from __future__ import print_function
import tensorflow as tf
import tensorlayer as tl
import numpy as np
import os, time, model, csv, random, gc, pickle
import nibabel as nib
import scannerpy

DATASET_DIR = "/root/shared/u-net-brain-tumor/dataset/"
MODEL_NPZ = DATASET_DIR + "model/u_net_all.npz"
INPUT_DIR = DATASET_DIR + "input/"
OUTPUT_DIR = DATASET_DIR + "output/"
[BATCH_SIZE, NW, NH, NZ] = [5, 240, 240, 4]

class BratsKernel(scannerpy.Kernel):
    def __init__(self, config, protobufs):
        self.protobufs = protobufs
        print('Build session')
        #with tf.device('/cpu:0'):
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        #with tf.device('/gpu:0'): #<- remove it if you train on CPU or other GPU
        ###======================== DEFIINE MODEL =======================###
        ## nz is 4 as we input all Flair, T1, T1c and T2.
        self.t_image = tf.placeholder('float32', [BATCH_SIZE, NW, NH, NZ], name='input_image')
        ## labels are either 0 or 1
        self.t_seg = tf.placeholder('float32', [BATCH_SIZE, NW, NH, 1], name='target_segment')
        ## test inference
        self.net = model.u_net(self.t_image, is_train=False, reuse=None, n_out=1)

        ###======================== DEFINE LOSS =========================###
        ## test losses
        self.test_out_seg = self.net.outputs
        self.test_dice_loss = 1 - tl.cost.dice_coe(self.test_out_seg, self.t_seg, axis=[0,1,2,3])#, 'jaccard', epsilon=1e-5)
        self.test_iou_loss = tl.cost.iou_coe(self.test_out_seg, self.t_seg, axis=[0,1,2,3])
        self.test_dice_hard = tl.cost.dice_hard_coe(self.test_out_seg, self.t_seg, axis=[0,1,2,3])

        ###======================== LOAD MODEL ==============================###
        tl.layers.initialize_global_variables(self.sess)
        ## load existing model if possible
        tl.files.load_and_assign_npz(sess=self.sess, name=MODEL_NPZ, network=self.net)

    def close(self):
        pass

    def vis_imgs(self, X, y, path):
        """ show one slice """
        if y.ndim == 2:
            y = y[:,:,np.newaxis]
        assert X.ndim == 3
        tl.vis.save_images(np.asarray([X[:,:,0,np.newaxis],
            X[:,:,1,np.newaxis], X[:,:,2,np.newaxis],
            X[:,:,3,np.newaxis], y]), size=(1, 5),
            image_path=path)

    def vis_imgs2(self, X, y_, y, path):
        """ show one slice with target """
        if y.ndim == 2:
            y = y[:,:,np.newaxis]
        if y_.ndim == 2:
            y_ = y_[:,:,np.newaxis]
        assert X.ndim == 3
        tl.vis.save_images(np.asarray([X[:,:,0,np.newaxis],
            X[:,:,1,np.newaxis], X[:,:,2,np.newaxis],
            X[:,:,3,np.newaxis], y_, y]), size=(1, 6),
            image_path=path)

    def read_img_dir(self, img_input_dir):
        img_name_list = [os.path.basename(os.path.normpath(img_input_dir))]
        print("img_name_list: {}".format(img_name_list))
        data_types = ['flair', 't1', 't1ce', 't2']
        #data_types_mean_std_dict = {i: {'mean': 0.0, 'std': 1.0} for i in data_types}

        with open(INPUT_DIR + 'mean_std_dict.pickle', 'rb') as f:
            data_types_mean_std_dict = pickle.load(f)
        print(data_types_mean_std_dict)

        ## GET NORMALIZE IMAGES
        X_dev_input = []
        X_dev_target = []

        for i in img_name_list:
            print("X_dev_input: {}, y_dev_target: {}".format(len(X_dev_input), len(X_dev_target)))
            all_3d_data = []
            for j in data_types:
                img_path = os.path.join(INPUT_DIR, i, i + '_' + j + '.nii.gz')
                img = nib.load(img_path).get_data()
                img = (img - data_types_mean_std_dict[j]['mean']) / data_types_mean_std_dict[j]['std']
                img = img.astype(np.float32)
                all_3d_data.append(img)

            seg_path = os.path.join(INPUT_DIR, i, i + '_seg.nii.gz')
            seg_img = nib.load(seg_path).get_data()
            seg_img = np.transpose(seg_img, (1, 0, 2))
            print("all_3d_data shape: {}".format(all_3d_data[0].shape[2]))
            for j in range(all_3d_data[0].shape[2]):
                combined_array = np.stack((all_3d_data[0][:, :, j], all_3d_data[1][:, :, j], all_3d_data[2][:, :, j], all_3d_data[3][:, :, j]), axis=2)
                combined_array = np.transpose(combined_array, (1, 0, 2))#.tolist()
                combined_array.astype(np.float32)
                X_dev_input.append(combined_array)

                seg_2d = seg_img[:, :, j]
                seg_2d.astype(int)
                X_dev_target.append(seg_2d)
            del all_3d_data
            gc.collect()
            print("finished {}".format(i))
            print("X_dev_input: {}, y_dev_target: {}".format(len(X_dev_input), len(X_dev_target)))

        X_dev_input = np.asarray(X_dev_input, dtype=np.float32)
        X_dev_target = np.asarray(X_dev_target)#, dtype=np.float32)
        return [X_dev_input, X_dev_target]

    def run_inference(self, X_test, y_test, img_output_dir):
        n_batch = 0
        total_dice, total_iou, total_dice_hard, n_batch = 0, 0, 0, 0
        output_once, output_done = False, False

        print("X_test: {}, y_test: {}".format(len(X_test), len(y_test)))
        for batch in tl.iterate.minibatches(inputs=X_test, targets=y_test,
                                        batch_size=BATCH_SIZE, shuffle=False):
            ## run inference on a batch
            b_images, b_labels = batch
            _dice, _iou, _diceh, out = self.sess.run([self.test_dice_loss,
                    self.test_iou_loss, self.test_dice_hard, self.net.outputs],
                    {self.t_image: b_images, self.t_seg: b_labels})
            total_dice += _dice; total_iou += _iou; total_dice_hard += _diceh

            ## save one prediction image per HGG/LGG input
            for i in range(BATCH_SIZE):
                if output_once and output_done:
                    output_done = False
                    break
                if np.max(b_images[i]) > 0:
                    output_png = img_output_dir + "/{}.png".format(n_batch)
                    self.vis_imgs2(b_images[i], b_labels[i], out[i], output_png)
                    print("saving output: {}".format(output_png))
                    output_done = True

            ## advance to the next batch
            n_batch += 1
            print(" **"+" "*17+"test 1-dice: %f hard-dice: %f iou: %f (2d no distortion)" %
                    (total_dice/n_batch, total_dice_hard/n_batch, total_iou/n_batch))

    def execute(self, input_columns):
        img_input_dir = input_columns[0]
        img_output_dir = input_columns[1]
        print("img_input_dir: {}, img_output_dir: {}".format(img_input_dir, img_output_dir)) 
        [X_dev_input, X_dev_target] = self.read_img_dir(img_input_dir)
        self.run_inference(X_dev_input, X_dev_target[:,:,:,np.newaxis], img_output_dir)
        return ["0"]

KERNEL = BratsKernel
