from attr import attrs, attrib, evolve
import subprocess as sp
import time
import json
import yaml
import tempfile
import argparse
import scannerpy
import os
import signal
import shlex
import cloudpickle
import base64
import math
import traceback
from abc import ABC
from threading import Thread, Condition
import logging as log
from scanner.metadata_pb2 import MachineParameters
from scannerpy._python import default_machine_params
import re
from datetime import datetime

MASTER_POOL = 'default-pool'
WORKER_POOL = 'workers'

log.basicConfig(level=log.INFO)

def run(s, detach=False):
    if detach:
        sp.Popen(s, shell=True, stdout=sp.PIPE, stderr=sp.PIPE)
        return None
    else:
        return sp.check_output(s, shell=True).decode('utf-8').strip()


@attrs(frozen=True)
class CloudConfig:
    project = attrib(type=str)
    service_key = attrib(type=str)
    storage_key_id = attrib(type=str)
    storage_key_secret = attrib(type=str)

    @service_key.default
    def _service_key_default(self):
        return os.environ['GOOGLE_APPLICATION_CREDENTIALS']

    @storage_key_id.default
    def _storage_key_id_default(self):
        return os.environ['AWS_ACCESS_KEY_ID']

    @storage_key_secret.default
    def _storage_key_secret_default(self):
        return os.environ['AWS_SECRET_ACCESS_KEY']


class MachineType(ABC):
    def get_cpu(self):
        raise NotImplemented

    def get_mem(self):
        raise NotImplemented

    def get_name(self):
        raise NotImplemented


GCP_MEM_RATIOS = {'standard': 3.75, 'highmem': 6.5, 'highcpu': 0.9}


@attrs(frozen=True)
class MachineTypeName(MachineType):
    name = attrib(type=str)

    def get_cpu(self):
        return int(self.name.split('-')[2])

    def get_mem(self):
        ty = self.name.split('-')[1]
        return int(GCP_MEM_RATIOS[ty] * self.get_cpu())

    def get_name(self):
        return self.name


@attrs(frozen=True)
class MachineTypeQuantity(MachineType):
    cpu = attrib(type=int)
    mem = attrib(type=int)

    def get_cpu(self):
        return self.cpu

    def get_mem(self):
        return self.mem

    def get_name(self):
        # See Google Cloud documentation for instance names.
        # https://cloud.google.com/compute/pricing#machinetype
        name = None
        mem_cpu_ratio = float(self.mem) / self.cpu
        if math.log2(self.cpu).is_integer():
            for k, ratio in GCP_MEM_RATIOS.items():
                if math.isclose(mem_cpu_ratio, ratio):
                    name = 'n1-{}-{}'.format(k, self.cpu)

        if name is None:
            name = 'custom-{}-{}'.format(self.cpu, self.mem * 1024)
            if mem_cpu_ratio > 6.5:
                name += '-ext'

        return name


@attrs(frozen=True)
class MachineConfig:
    image = attrib(type=str)
    type = attrib(type=MachineType)
    disk = attrib(type=int)
    gpu = attrib(type=int, default=0)
    gpu_type = attrib(type=str, default='nvidia-tesla-p100')
    preemptible = attrib(type=bool, default=False)

    def price(self):
        parts = self.type.get_name().split('-')
        category = parts[0]
        preempt = 'preemptible' if self.preemptible else 'normal'
        p = 0

        if category == 'n1':
            prices = {
                'normal': {
                    'standard': 0.0475,
                    'highmem': 0.1184 / 2,
                    'highcpu': 0.0709 / 2
                },
                'preemptible': {
                    'standard': 0.01,
                    'highmem': 0.0125,
                    'highcpu': 0.0075
                }
            }
            p = prices[preempt][parts[1]] * int(parts[2])

        elif category == 'custom':
            [cpu, mem] = parts[1:3]
            cpu = int(cpu)
            mem = int(mem)
            prices = {
                'normal': {
                    'cpu': 0.033174,
                    'mem': 0.004446,
                },
                'preemptible': {
                    'cpu': 0.00698,
                    'mem': 0.00094
                }
            }
            p = prices[preempt]['cpu'] * cpu + prices[preempt]['mem'] * (mem / 1024)

        else:
            raise Exception('Invalid category {}'.format(category))

        gpu_prices = {
            'normal': {
                'nvidia-tesla-k80': 0.45,
                'nvidia-tesla-p100': 1.46,
                'nvidia-tesla-v100': 2.48
            },
            'preemptible': {
                'nvidia-tesla-k80': 0.135,
                'nvidia-tesla-p100': 0.43,
                'nvidia-tesla-v100': 0.74
            }
        }

        p += gpu_prices[preempt][self.gpu_type] * self.gpu
        return p


@attrs(frozen=True)
class ClusterConfig:
    id = attrib(type=str)
    num_workers = attrib(type=int)
    master = attrib(type=MachineConfig)
    worker = attrib(type=MachineConfig)
    zone = attrib(type=str, default='us-east1-b')
    kube_version = attrib(type=str, default='latest')
    num_load_workers = attrib(type=int, default=8)
    num_save_workers = attrib(type=int, default=8)
    autoscale = attrib(type=bool, default=True)
    no_workers_timeout = attrib(type=int, default=600)
    scopes = attrib(
        type=frozenset,
        default=frozenset([
            "https://www.googleapis.com/auth/compute",
            "https://www.googleapis.com/auth/devstorage.read_write",
            "https://www.googleapis.com/auth/logging.write",
            "https://www.googleapis.com/auth/monitoring", "https://www.googleapis.com/auth/pubsub",
            "https://www.googleapis.com/auth/servicecontrol",
            "https://www.googleapis.com/auth/service.management.readonly",
            "https://www.googleapis.com/auth/trace.append"
        ]))
    scanner_config = attrib(
        type=str, default=os.path.join(os.environ['HOME'], '.scanner/config.toml'))
    pipelines = attrib(type=frozenset, default=frozenset([]))

    # unit is $/hr
    def price(self, no_master=False):
        return (self.master.price() if not no_master else 0) + self.worker.price() * self.num_workers


class Cluster:
    def __init__(self, cloud_config, cluster_config, no_start=False, no_delete=False, containers=None):
        self._cloud_config = cloud_config
        self._cluster_config = cluster_config
        self._cluster_cmd = 'gcloud container --project {} clusters --zone {}' \
            .format(self._cloud_config.project, self._cluster_config.zone)
        self._no_start = no_start
        self._no_delete = no_delete
        self._containers = containers or []

    def config(self):
        return self._cluster_config

    def get_kube_info(self, kind, namespace='default'):
        return json.loads(run('kubectl get {} -o json -n {}'.format(kind, namespace)))

    def get_by_owner(self, ty, owner, namespace='default'):
        return run(
            'kubectl get {} -o json -n {} | jq -r \'.items[] | select(.metadata.ownerReferences[0].name == "{}") | .metadata.name\'' \
            .format(ty, namespace, owner))

    def get_object(self, info, name):
        for item in info['items']:
            if item['metadata']['name'] == name:
                return item
        return None

    def get_pod(self, deployment, namespace='default'):
        while True:
            rs = self.get_by_owner('rs', deployment, namespace)
            pod_name = self.get_by_owner('pod', rs, namespace)
            if "\n" not in pod_name and pod_name != "":
                break
            time.sleep(1)

        while True:
            pod = self.get_object(self.get_kube_info('pod', namespace), pod_name)
            if pod is not None:
                return pod
            time.sleep(1)

    def running(self, pool=MASTER_POOL):
        return run(
            '{cmd} list --cluster={id} --format=json | jq \'.[] | select(.name == "{pool}")\''.
            format(
                cmd=self._cluster_cmd.replace('clusters', 'node-pools'),
                id=self._cluster_config.id,
                pool=pool)) != ''

    def get_credentials(self):
        if self._cluster_config.id not in run(
                'kubectl config view -o json | jq \'.["current-context"]\' -r'):
            run('{cmd} get-credentials {id}'.format(
                cmd=self._cluster_cmd, id=self._cluster_config.id))

    def create_object(self, template):
        with tempfile.NamedTemporaryFile() as f:
            f.write(yaml.dump(template).encode())
            f.flush()
            run('kubectl create -f {}'.format(f.name))

    def make_container(self, name, machine_config):
        template = {
            'name': name,
            'image': machine_config.image,
            'command': ['/bin/bash'],
            'args': ['-c', 'python3 -c "from scannerpy import kube; kube.{}()"'.format(name)],
            'imagePullPolicy': 'Always',
            'volumeMounts': [{
                'name': 'service-key',
                'mountPath': '/secret'
            }, {
                'name': 'scanner-config',
                'mountPath': '/root/.scanner/config.toml',
                'subPath': 'config.toml'
            }],
            'env': [
                {'name': 'GOOGLE_APPLICATION_CREDENTIALS',
                 'value': '/secret/{}'.format(os.path.basename(self._cloud_config.service_key))},
                {'name': 'AWS_ACCESS_KEY_ID',
                 'valueFrom': {'secretKeyRef': {
                     'name': 'aws-storage-key',
                     'key': 'AWS_ACCESS_KEY_ID'
                 }}},
                {'name': 'AWS_SECRET_ACCESS_KEY',
                 'valueFrom': {'secretKeyRef': {
                     'name': 'aws-storage-key',
                     'key': 'AWS_SECRET_ACCESS_KEY'
                 }}},
                {'name': 'NO_WORKERS_TIMEOUT',
                 'value': str(self._cluster_config.no_workers_timeout)},
                {'name': 'GLOG_minloglevel',
                 'value': '0'},
                {'name': 'GLOG_logtostderr',
                 'value': '1'},
                {'name': 'GLOG_v',
                 'value': '2' if name == 'master' else '1'},
                {'name': 'NUM_LOAD_WORKERS',
                 'value': str(self._cluster_config.num_load_workers)},
                {'name': 'NUM_SAVE_WORKERS',
                 'value': str(self._cluster_config.num_save_workers)},
                {'name': 'PIPELINES',
                 'value': base64.b64encode(cloudpickle.dumps(self._cluster_config.pipelines))},
                # HACK(wcrichto): GPU decode for interlaced videos is broken, so forcing CPU
                # decode instead for now.
                {'name': 'FORCE_CPU_DECODE',
                 'value': '1'}
            ],
            'resources': {},
            'securityContext': {'capabilities': {
                'add': ['SYS_PTRACE']  # Allows gdb to work in container
            }}
        }  # yapf: disable
        if name == 'master':
            template['ports'] = [{
                'containerPort': 8080,
            }]

        if machine_config.gpu > 0:
            template['resources']['limits'] = {'nvidia.com/gpu': machine_config.gpu}
        else:
            if name == 'worker':
                template['resources']['requests'] = {
                    'cpu': machine_config.type.get_cpu() / 2.0 + 0.1
                }

        return template

    def make_deployment(self, name, machine_config, replicas):
        template = {
            'apiVersion': 'apps/v1beta1',
            'kind': 'Deployment',
            'metadata': {'name': 'scanner-{}'.format(name)},
            'spec': {  # DeploymentSpec
                'replicas': replicas,
                'template': {
                    'metadata': {'labels': {'app': 'scanner-{}'.format(name)}},
                    'spec': {  # PodSpec
                        'containers': [self.make_container(name, machine_config)] + self._containers,
                        'volumes': [{
                            'name': 'service-key',
                            'secret': {
                                'secretName': 'service-key',
                                'items': [{
                                    'key': os.path.basename(self._cloud_config.service_key),
                                    'path': os.path.basename(self._cloud_config.service_key)
                                }]
                            }
                        }, {
                            'name': 'scanner-config',
                            'configMap': {'name': 'scanner-config'}
                        }],
                        'nodeSelector': {
                            'cloud.google.com/gke-nodepool':
                            MASTER_POOL if name == 'master' else WORKER_POOL
                        }
                    }
                }
            }
        }  # yapf: disable

        return template

    def _cluster_start(self):
        cfg = self._cluster_config
        fmt_args = {
            'cmd': self._cluster_cmd,
            'cluster_id': cfg.id,
            'cluster_version': cfg.kube_version,
            'master_machine': cfg.master.type.get_name(),
            'master_disk': cfg.master.disk,
            'worker_machine': cfg.worker.type.get_name(),
            'worker_disk': cfg.worker.disk,
            'scopes': ','.join(cfg.scopes),
            'initial_size': cfg.num_workers,
            'accelerator': '--accelerator type={},count={}'.format(cfg.worker.gpu_type, cfg.worker.gpu) if cfg.worker.gpu > 0 else '',
            'preemptible': '--preemptible' if cfg.worker.preemptible else '',
            'autoscaling': '--enable-autoscaling --min-nodes 0 --max-nodes {}'.format(cfg.num_workers) if cfg.autoscale else '',
            'master_cpu_platform': '--min-cpu-platform skylake' if cfg.master.type.get_cpu() > 64 else '',
            'worker_cpu_platform': '--min-cpu-platform skylake' if cfg.worker.type.get_cpu() > 64 else ''
        }  # yapf: disable


        # wcrichto 2-22-19: after correspondence with Edward Doan at GCP on cluster startup times,
        # they say that enabling autoscaling on the master node pool should reduce startup times for worker pool
        cluster_cmd = """
{cmd} -q create "{cluster_id}" \
        --cluster-version "{cluster_version}" \
        --machine-type "{master_machine}" \
        --image-type "COS" \
        --disk-size "{master_disk}" \
        --scopes {scopes} \
        --num-nodes "1" \
        --enable-cloud-logging \
        --enable-autoscaling --min-nodes 1 --max-nodes 1 \
        {accelerator} \
        {master_cpu_platform}
        """.format(**fmt_args)

        if not self.running(pool=MASTER_POOL):
            try:
                log.info('Cluster price: ${:.2f}/hr'.format(cfg.price()))
            except Exception:
                log.info('Failed to compute cluster price with error:')
                traceback.print_exc()

            log.info('Creating master...')
            run(cluster_cmd)
            log.info(
                'https://console.cloud.google.com/kubernetes/clusters/details/{zone}/{cluster_id}?project={project}&tab=details' \
                .format(zone=cfg.zone, project=self._cloud_config.project, **fmt_args))

            fmt_args['cmd'] = fmt_args['cmd'].replace('clusters', 'node-pools')
            pool_cmd = """
    {cmd} -q create workers \
            --cluster "{cluster_id}" \
            --machine-type "{worker_machine}" \
            --image-type "COS" \
            --disk-size "{worker_disk}" \
            --scopes {scopes} \
            --num-nodes "{initial_size}" \
            {autoscaling} \
            {preemptible} \
            {accelerator} \
            {worker_cpu_platform}
            """.format(**fmt_args)

            log.info('Creating workers...')
            try:
                run(pool_cmd)
            except sp.CalledProcessError as e:
                # Ignore errors for now due to GKE issue
                log.error('Worker pool command errored: {}'.format(e))

            # Wait for cluster to enter reconciliation if it's going to occur
            log.info('Waiting for cluster to reconcile...')
            if cfg.num_workers > 1:
                time.sleep(60)

            # If we requested workers up front, we have to wait for the cluster to reconcile while
            # they are being allocated
            while True:
                cluster_status = run(
                    '{cmd} list --format=json | jq -r \'.[] | select(.name == "{id}") | .status\''.
                    format(cmd=self._cluster_cmd, id=cfg.id))

                if cluster_status == 'RECONCILING':
                    time.sleep(5)
                else:
                    if cluster_status != 'RUNNING':
                        raise Exception(
                            'Expected cluster status RUNNING, got: {}'.format(cluster_status))
                    break

        # TODO(wcrichto): run this if GPU driver daemon isn't running yet
        if cfg.worker.gpu > 0:
            # Install GPU drivers
            # https://cloud.google.com/kubernetes-engine/docs/concepts/gpus#installing_drivers
            run('kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/stable/nvidia-driver-installer/cos/daemonset-preloaded.yaml'
                )

    def _kube_start(self, reset=True, wait=True):
        cfg = self._cluster_config
        deploy = self.get_object(self.get_kube_info('deployments'), 'scanner-worker')
        if deploy is not None:
            num_workers = deploy['status']['replicas']
        else:
            num_workers = cfg.num_workers

        if reset:
            log.info('Deleting current deployments...')
            run('kubectl delete deploy/scanner-master deploy/scanner-worker --ignore-not-found=true'
                )
            run('kubectl delete service/scanner-master --ignore-not-found=true')

        secrets = self.get_kube_info('secrets')
        if self.get_object(secrets, 'service-key') is None:
            log.info('Making secrets...')
            run('kubectl create secret generic service-key --from-file={}' \
                .format(self._cloud_config.service_key))

        if self.get_object(secrets, 'aws-storage-key') is None:
            run('kubectl create secret generic aws-storage-key --from-literal=AWS_ACCESS_KEY_ID={} --from-literal=AWS_SECRET_ACCESS_KEY={}' \
                .format(self._cloud_config.storage_key_id, self._cloud_config.storage_key_secret))

        configmaps = self.get_kube_info('configmaps')
        if self.get_object(configmaps, 'scanner-config') is None:
            run('kubectl create configmap scanner-config --from-file={}' \
                .format(self._cluster_config.scanner_config))

        deployments = self.get_kube_info('deployments')
        if self.get_object(deployments, 'scanner-master') is None:
            log.info('Creating deployments...')
            self.create_object(self.make_deployment('master', self._cluster_config.master, 1))

        services = self.get_kube_info('services')
        if self.get_object(services, 'scanner-master') is None:
            run('kubectl expose deploy/scanner-master --type=NodePort --port=8080')

        if self.get_object(deployments, 'scanner-worker') is None:
            self.create_object(
                self.make_deployment('worker', self._cluster_config.worker, num_workers))

        if wait:
            log.info('Waiting on master...')
            while True:
                master = self.get_pod('scanner-master')
                if master['status']['phase'] == 'Running':
                    break
                time.sleep(1.0)

    def start(self, reset=True, wait=True):
        self._cluster_start()
        self.get_credentials()
        self._kube_start(reset, wait)
        log.info('Finished startup.')

    def resize(self, size):
        log.info('Resized cluster price: ${:.2f}/hr'.format(evolve(self._cluster_config, num_workers=size).price()))

        log.info('Resizing cluster...')
        if not self._cluster_config.autoscale:
            run('{cmd} resize {id} -q --node-pool=workers --size={size}' \
                .format(cmd=self._cluster_cmd, id=self._cluster_config.id, size=size))
        else:
            run('{cmd} update {id} -q --node-pool=workers --enable-autoscaling --max-nodes={size}' \
                .format(cmd=self._cluster_cmd, id=self._cluster_config.id, size=size))

        log.info('Scaling deployment...')
        run('kubectl scale deploy/scanner-worker --replicas={}'.format(size))

    def delete(self, prompt=False):
        run('{cmd} {prompt} delete {id}'.format(
            cmd=self._cluster_cmd, prompt='-q' if not prompt else '', id=self._cluster_config.id))

    def master_address(self):
        ip = run('''
            kubectl get pods -l 'app=scanner-master' -o json | \
            jq '.items[0].spec.nodeName' -r | \
            xargs -I {} kubectl get nodes/{} -o json | \
            jq '.status.addresses[] | select(.type == "ExternalIP") | .address' -r
            ''')

        port = run('''
            kubectl get svc/scanner-master -o json | \
            jq '.spec.ports[0].nodePort' -r
            ''')

        return '{}:{}'.format(ip, port)

    def client(self, retries=3, **kwargs):
        while True:
            try:
                return scannerpy.Client(
                    master=self.master_address(), start_cluster=False, **kwargs)
            except scannerpy.ScannerException:
                if retries == 0:
                    raise
                else:
                    retries -= 1

    def job_status(self):
        sc = self.client()
        jobs = sc.get_active_jobs()
        if len(jobs) > 0:
            sc.wait_on_job(jobs[0])

    def healthy(self):
        failed = run('''
            kubectl get pod -o json | \
            jq -r '.items[].status | select(.phase == "Failed" or .containerStatuses[0].lastState.terminated.reason == "OOMKilled")'
            ''')

        return failed == ''

    def monitor(self, sc):
        done = None
        done_cvar = Condition()

        def loop_set(cond, val):
            def wrapper():
                nonlocal done
                nonlocal done_cvar

                while True:
                    if done is not None:
                        break

                    if cond():
                        with done_cvar:
                            done = val
                            done_cvar.notify()

                    time.sleep(1.0)

            return wrapper

        jobs = sc.get_active_jobs()
        if len(jobs) == 0:
            raise Exception("No active jobs")

        gen = sc.wait_on_job_gen(jobs[0])

        def scanner_check():
            nonlocal gen
            try:
                next(gen)
                return False
            except StopIteration:
                return True

        def health_check():
            return not self.healthy()

        metrics = []

        def resource_check():
            nonlocal metrics
            metrics.extend([{'TIME': datetime.now(), **r} for r in self.resource_metrics()])
            return False

        checks = [(scanner_check, True), (health_check, False), (resource_check, None)]

        threads = [Thread(target=loop_set(f, v), daemon=True) for (f, v) in checks]
        for t in threads:
            t.start()

        with done_cvar:
            while done is None:
                done_cvar.wait()

        for t in threads:
            t.join()

        return done, metrics

    def resource_metrics(self):
        table = run('kubectl top nodes').split('\n')
        (header, rows) = (table[0], table[1:])

        def match(line):
            return re.findall(r'([^\s]+)\s*', line)

        columns = match(header)
        values = [{
            c: int(re.search(r'(\d+)', v).group(1)) if c != 'NAME' else v
            for (c, v) in zip(columns, match(row))
        } for row in rows]
        values = [v for v in values if 'default-pool' not in v['NAME']]

        return values

    def master_logs(self, previous=False):
        master = self.get_pod('scanner-master')
        print(run('kubectl logs pod/{} master {}'.format(master['metadata']['name'], '--previous' if previous else '')))

    def worker_logs(self, n, previous=False):
        workers = [pod for pod in self.get_kube_info('pod')['items'] if pod['metadata']['labels']['app'] == 'scanner-worker']
        print(run('kubectl logs pod/{} worker {}'.format(workers[n]['metadata']['name'], '--previous' if previous else '')))

    def trace(self, path, subsample=None, job=None):
        self.get_credentials()

        sc = self.client()

        # Get most recent job id if none is provided
        if job is None:
            log.info('Fetching job ID')
            job = max([
                int(line.split('/')[-2])
                for line in sp.check_output('gsutil ls gs://{}/{}/jobs'.format(
                        sc.config.config['storage']['bucket'],
                        sc.config.db_path),
                    shell=True) \
                .decode('utf-8').split('\n')[:-1]
            ])

        log.info('Writing trace...')
        sc.profiler(job, subsample=subsample).write_trace(path)
        log.info('Trace written.')

    def __enter__(self):
        if not self._no_start:
            self.start()
        return self

    def __exit__(self, *args, **kwargs):
        if not self._no_delete:
            self.delete()

    def cli(self):
        parser = argparse.ArgumentParser()
        command = parser.add_subparsers(dest='command')
        command.required = True
        create = command.add_parser('start', help='Create cluster')
        create.add_argument(
            '--no-reset', '-nr', action='store_true', help='Delete current deployments')
        create.add_argument('--no-wait', '-nw', action='store_true', help='Don\'t wait for master')
        create.add_argument(
            '--num-workers', '-n', type=int, default=1, help='Initial number of workers')
        delete = command.add_parser('delete', help='Delete cluster')
        delete.add_argument(
            '--no-prompt', '-np', action='store_true', help='Don\'t prompt for deletion')
        resize = command.add_parser('resize', help='Resize number of nodes in cluster')
        resize.add_argument('size', type=int, help='Number of nodes')
        command.add_parser('get-credentials', help='Setup kubectl with credentials')
        command.add_parser('job-status', help='View status of current running job')
        master_logs = command.add_parser('master-logs', help='Get logs of Scanner master')
        master_logs.add_argument('--previous', '-p', action='store_true', help='Get logs for previous container')
        worker_logs = command.add_parser('worker-logs', help='Get logs of a Scanner worker')
        worker_logs.add_argument('n', type=int, help='Index of worker')
        worker_logs.add_argument('--previous', '-p', action='store_true', help='Get logs for previous container')
        trace = command.add_parser('trace', help='Extract profiler trace')
        trace.add_argument('path', help='Path to output trace')
        trace.add_argument('--subsample', type=int, help='Number of workers to include in trace')
        trace.add_argument('--job', type=int, help='Job ID to extract (default is latest)')

        args = parser.parse_args()
        if args.command == 'start':
            self.start(reset=not args.no_reset, wait=not args.no_wait)

        elif args.command == 'delete':
            self.delete(prompt=not args.no_prompt)

        elif args.command == 'resize':
            self.resize(args.size)

        elif args.command == 'get-credentials':
            self.get_credentials()

        elif args.command == 'job-status':
            self.job_status()

        elif args.command == 'master-logs':
            self.master_logs(previous=args.previous)

        elif args.command == 'worker-logs':
            self.worker_logs(args.n, previous=args.previous)

        elif args.command == 'trace':
            self.trace(args.path, subsample=args.subsample, job=args.job)


def master():
    log.info('Scanner: starting master...')
    scannerpy.start_master(
        port='8080',
        block=True,
        watchdog=False,
        no_workers_timeout=int(os.environ['NO_WORKERS_TIMEOUT']))


def worker():
    machine_params = MachineParameters()
    machine_params.ParseFromString(default_machine_params())
    machine_params.num_load_workers = int(os.environ['NUM_LOAD_WORKERS'])
    machine_params.num_save_workers = int(os.environ['NUM_SAVE_WORKERS'])

    log.info('Scanner: starting worker...')
    scannerpy.start_worker(
        '{}:{}'.format(os.environ['SCANNER_MASTER_SERVICE_HOST'],
                       os.environ['SCANNER_MASTER_SERVICE_PORT']),
        machine_params=machine_params.SerializeToString(),
        block=True,
        watchdog=False,
        port=5002)
