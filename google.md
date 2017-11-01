# Getting started with Google Cloud

This guide will walk you through setting up Scanner on Google Cloud. You will need to have a Google account.

## 1. Install the Cloud SDK

On your local machine (laptop/desktop), follow the instructions here to install Google's Cloud SDK: [https://cloud.google.com/sdk/downloads](https://cloud.google.com/sdk/downloads)

## 2. Create a project

If you do not already have a project created, pick a project ID for your application, e.g. `my-scanner-project`. Then run:
```bash
gcloud projects create <project ID>
```

## 3. Make a bucket

You will need to store your videos in Google Cloud Storage. Cloud Storage is organized into independent buckets (like top-level directories). Pick a name for your bucket, e.g. `scanner-data`, and run:
```bash
gsutil mb gs://scanner-data
```

## 4. Set up your Scanner config

Change the storage heading in your `~/.scanner.toml` to use GCS:
```toml
[storage]
type = "gcs"
bucket = "<your bucket name>"
db_path = "scanner_db"
```

## 5. Upload your videos into your bucket

You can copy videos onto GCS like this:
```bash
gsutil cp example.mp4 gs://scanner-data/videos/
```

## 6. You're done!

Now, whenever you want to specify an ingest path, it does not need a leading slash and should not include the bucket name. For example, with the config above, the following is a valid ingest path:
```
videos/example.mp4
```

If you want to use Google Cloud to scale computation instead of just storage, take a look at our Kubernetes adapter: [https://github.com/scanner-research/scanner-kube](https://github.com/scanner-research/scanner-kube)
