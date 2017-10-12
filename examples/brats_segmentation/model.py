import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
import numpy as np


from tensorlayer.layers import *
def u_net(x, is_train=False, reuse=False, n_out=1):
    _, nx, ny, nz = x.get_shape().as_list()
    with tf.variable_scope("u_net", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        inputs = InputLayer(x, name='inputs')
        conv1 = Conv2d(inputs, 64, (3, 3), act=tf.nn.relu, name='conv1_1')
        conv1 = Conv2d(conv1, 64, (3, 3), act=tf.nn.relu, name='conv1_2')
        pool1 = MaxPool2d(conv1, (2, 2), name='pool1')
        conv2 = Conv2d(pool1, 128, (3, 3), act=tf.nn.relu, name='conv2_1')
        conv2 = Conv2d(conv2, 128, (3, 3), act=tf.nn.relu, name='conv2_2')
        pool2 = MaxPool2d(conv2, (2, 2), name='pool2')
        conv3 = Conv2d(pool2, 256, (3, 3), act=tf.nn.relu, name='conv3_1')
        conv3 = Conv2d(conv3, 256, (3, 3), act=tf.nn.relu, name='conv3_2')
        pool3 = MaxPool2d(conv3, (2, 2), name='pool3')
        conv4 = Conv2d(pool3, 512, (3, 3), act=tf.nn.relu, name='conv4_1')
        conv4 = Conv2d(conv4, 512, (3, 3), act=tf.nn.relu, name='conv4_2')
        pool4 = MaxPool2d(conv4, (2, 2), name='pool4')
        conv5 = Conv2d(pool4, 1024, (3, 3), act=tf.nn.relu, name='conv5_1')
        conv5 = Conv2d(conv5, 1024, (3, 3), act=tf.nn.relu, name='conv5_2')

        up4 = DeConv2d(conv5, 512, (3, 3), (nx/8, ny/8), (2, 2), name='deconv4')
        up4 = ConcatLayer([up4, conv4], 3, name='concat4')
        conv4 = Conv2d(up4, 512, (3, 3), act=tf.nn.relu, name='uconv4_1')
        conv4 = Conv2d(conv4, 512, (3, 3), act=tf.nn.relu, name='uconv4_2')
        up3 = DeConv2d(conv4, 256, (3, 3), (nx/4, ny/4), (2, 2), name='deconv3')
        up3 = ConcatLayer([up3, conv3], 3, name='concat3')
        conv3 = Conv2d(up3, 256, (3, 3), act=tf.nn.relu, name='uconv3_1')
        conv3 = Conv2d(conv3, 256, (3, 3), act=tf.nn.relu, name='uconv3_2')
        up2 = DeConv2d(conv3, 128, (3, 3), (nx/2, ny/2), (2, 2), name='deconv2')
        up2 = ConcatLayer([up2, conv2], 3, name='concat2')
        conv2 = Conv2d(up2, 128, (3, 3), act=tf.nn.relu,  name='uconv2_1')
        conv2 = Conv2d(conv2, 128, (3, 3), act=tf.nn.relu, name='uconv2_2')
        up1 = DeConv2d(conv2, 64, (3, 3), (nx/1, ny/1), (2, 2), name='deconv1')
        up1 = ConcatLayer([up1, conv1] , 3, name='concat1')
        conv1 = Conv2d(up1, 64, (3, 3), act=tf.nn.relu, name='uconv1_1')
        conv1 = Conv2d(conv1, 64, (3, 3), act=tf.nn.relu, name='uconv1_2')
        conv1 = Conv2d(conv1, n_out, (1, 1), act=tf.nn.sigmoid, name='uconv1')
    return conv1

# def u_net(x, is_train=False, reuse=False, pad='SAME', n_out=2):
#     """ Original U-Net for cell segmentataion
#     http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/
#     Original x is [batch_size, 572, 572, ?], pad is VALID
#     """
#     from tensorlayer.layers import InputLayer, Conv2d, MaxPool2d, DeConv2d, ConcatLayer
#     nx = int(x._shape[1])
#     ny = int(x._shape[2])
#     nz = int(x._shape[3])
#     print(" * Input: size of image: %d %d %d" % (nx, ny, nz))
#
#     w_init = tf.truncated_normal_initializer(stddev=0.01)
#     b_init = tf.constant_initializer(value=0.0)
#     with tf.variable_scope("u_net", reuse=reuse):
#         tl.layers.set_name_reuse(reuse)
#         inputs = InputLayer(x, name='inputs')
#
#         conv1 = Conv2d(inputs, 64, (3, 3), act=tf.nn.relu, padding=pad,
#                     W_init=w_init, b_init=b_init, name='conv1_1')
#         conv1 = Conv2d(conv1, 64, (3, 3), act=tf.nn.relu, padding=pad,
#                     W_init=w_init, b_init=b_init, name='conv1_2')
#         pool1 = MaxPool2d(conv1, (2, 2), padding=pad, name='pool1')
#
#         conv2 = Conv2d(pool1, 128, (3, 3), act=tf.nn.relu, padding=pad,
#                     W_init=w_init, b_init=b_init, name='conv2_1')
#         conv2 = Conv2d(conv2, 128, (3, 3), act=tf.nn.relu, padding=pad,
#                     W_init=w_init, b_init=b_init, name='conv2_2')
#         pool2 = MaxPool2d(conv2, (2, 2), padding=pad, name='pool2')
#
#         conv3 = Conv2d(pool2, 256, (3, 3), act=tf.nn.relu, padding=pad,
#                     W_init=w_init, b_init=b_init, name='conv3_1')
#         conv3 = Conv2d(conv3, 256, (3, 3), act=tf.nn.relu, padding=pad,
#                     W_init=w_init, b_init=b_init, name='conv3_2')
#         pool3 = MaxPool2d(conv3, (2, 2), padding=pad, name='pool3')
#
#         conv4 = Conv2d(pool3, 512, (3, 3), act=tf.nn.relu, padding=pad,
#                     W_init=w_init, b_init=b_init, name='conv4_1')
#         conv4 = Conv2d(conv4, 512, (3, 3), act=tf.nn.relu, padding=pad,
#                     W_init=w_init, b_init=b_init, name='conv4_2')
#         pool4 = MaxPool2d(conv4, (2, 2), padding=pad, name='pool4')
#
#         conv5 = Conv2d(pool4, 1024, (3, 3), act=tf.nn.relu, padding=pad,
#                     W_init=w_init, b_init=b_init, name='conv5_1')
#         conv5 = Conv2d(conv5, 1024, (3, 3), act=tf.nn.relu, padding=pad,
#                     W_init=w_init, b_init=b_init, name='conv5_2')
#
#         print(" * After conv: %s" % conv5.outputs)
#
#         up4 = DeConv2d(conv5, 512, (3, 3), out_size = (nx/8, ny/8),
#                     strides=(2, 2), padding=pad, act=None,
#                     W_init=w_init, b_init=b_init, name='deconv4')
#         up4 = ConcatLayer([up4, conv4], concat_dim=3, name='concat4')
#         conv4 = Conv2d(up4, 512, (3, 3), act=tf.nn.relu, padding=pad,
#                     W_init=w_init, b_init=b_init, name='uconv4_1')
#         conv4 = Conv2d(conv4, 512, (3, 3), act=tf.nn.relu, padding=pad,
#                     W_init=w_init, b_init=b_init, name='uconv4_2')
#
#         up3 = DeConv2d(conv4, 256, (3, 3), out_size = (nx/4, ny/4),
#                     strides=(2, 2), padding=pad, act=None,
#                     W_init=w_init, b_init=b_init, name='deconv3')
#         up3 = ConcatLayer([up3, conv3], concat_dim=3, name='concat3')
#         conv3 = Conv2d(up3, 256, (3, 3), act=tf.nn.relu, padding=pad,
#                     W_init=w_init, b_init=b_init, name='uconv3_1')
#         conv3 = Conv2d(conv3, 256, (3, 3), act=tf.nn.relu, padding=pad,
#                     W_init=w_init, b_init=b_init, name='uconv3_2')
#
#         up2 = DeConv2d(conv3, 128, (3, 3), out_size=(nx/2, ny/2),
#                     strides=(2, 2), padding=pad, act=None,
#                     W_init=w_init, b_init=b_init, name='deconv2')
#         up2 = ConcatLayer([up2, conv2] ,concat_dim=3, name='concat2')
#         conv2 = Conv2d(up2, 128, (3, 3), act=tf.nn.relu, padding=pad,
#                     W_init=w_init, b_init=b_init, name='uconv2_1')
#         conv2 = Conv2d(conv2, 128, (3, 3), act=tf.nn.relu, padding=pad,
#                     W_init=w_init, b_init=b_init, name='uconv2_2')
#
#         up1 = DeConv2d(conv2, 64, (3, 3), out_size=(nx/1, ny/1),
#                     strides=(2, 2), padding=pad, act=None,
#                     W_init=w_init, b_init=b_init, name='deconv1')
#         up1 = ConcatLayer([up1, conv1] ,concat_dim=3, name='concat1')
#         conv1 = Conv2d(up1, 64, (3, 3), act=tf.nn.relu, padding=pad,
#                     W_init=w_init, b_init=b_init, name='uconv1_1')
#         conv1 = Conv2d(conv1, 64, (3, 3), act=tf.nn.relu, padding=pad,
#                     W_init=w_init, b_init=b_init, name='uconv1_2')
#
#         conv1 = Conv2d(conv1, n_out, (1, 1), act=tf.nn.sigmoid, name='uconv1')
#         print(" * Output: %s" % conv1.outputs)
#
#         # logits0 = conv1.outputs[:,:,:,0]            # segmentataion
#         # logits1 = conv1.outputs[:,:,:,1]            # edge
#         # logits0 = tf.expand_dims(logits0, axis=3)
#         # logits1 = tf.expand_dims(logits1, axis=3)
#     return conv1


def u_net_bn(x, is_train=False, reuse=False, batch_size=None, pad='SAME', n_out=1):
    """image to image translation via conditional adversarial learning"""
    nx = int(x._shape[1])
    ny = int(x._shape[2])
    nz = int(x._shape[3])
    print(" * Input: size of image: %d %d %d" % (nx, ny, nz))

    w_init = tf.truncated_normal_initializer(stddev=0.01)
    b_init = tf.constant_initializer(value=0.0)
    gamma_init=tf.random_normal_initializer(1., 0.02)
    with tf.variable_scope("u_net", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        inputs = InputLayer(x, name='inputs')

        conv1 = Conv2d(inputs, 64, (4, 4), (2, 2), act=None, padding=pad, W_init=w_init, b_init=b_init, name='conv1')
        conv2 = Conv2d(conv1, 128, (4, 4), (2, 2), act=None, padding=pad, W_init=w_init, b_init=b_init, name='conv2')
        conv2 = BatchNormLayer(conv2, act=lambda x: tl.act.lrelu(x, 0.2), is_train=is_train, gamma_init=gamma_init, name='bn2')

        conv3 = Conv2d(conv2, 256, (4, 4), (2, 2), act=None, padding=pad, W_init=w_init, b_init=b_init, name='conv3')
        conv3 = BatchNormLayer(conv3, act=lambda x: tl.act.lrelu(x, 0.2), is_train=is_train, gamma_init=gamma_init, name='bn3')

        conv4 = Conv2d(conv3, 512, (4, 4), (2, 2), act=None, padding=pad, W_init=w_init, b_init=b_init, name='conv4')
        conv4 = BatchNormLayer(conv4, act=lambda x: tl.act.lrelu(x, 0.2), is_train=is_train, gamma_init=gamma_init, name='bn4')

        conv5 = Conv2d(conv4, 512, (4, 4), (2, 2), act=None, padding=pad, W_init=w_init, b_init=b_init, name='conv5')
        conv5 = BatchNormLayer(conv5, act=lambda x: tl.act.lrelu(x, 0.2), is_train=is_train, gamma_init=gamma_init, name='bn5')

        conv6 = Conv2d(conv5, 512, (4, 4), (2, 2), act=None, padding=pad, W_init=w_init, b_init=b_init, name='conv6')
        conv6 = BatchNormLayer(conv6, act=lambda x: tl.act.lrelu(x, 0.2), is_train=is_train, gamma_init=gamma_init, name='bn6')

        conv7 = Conv2d(conv6, 512, (4, 4), (2, 2), act=None, padding=pad, W_init=w_init, b_init=b_init, name='conv7')
        conv7 = BatchNormLayer(conv7, act=lambda x: tl.act.lrelu(x, 0.2), is_train=is_train, gamma_init=gamma_init, name='bn7')

        conv8 = Conv2d(conv7, 512, (4, 4), (2, 2), act=lambda x: tl.act.lrelu(x, 0.2), padding=pad, W_init=w_init, b_init=b_init, name='conv8')
        print(" * After conv: %s" % conv8.outputs)
        # exit()
        # print(nx/8)
        up7 = DeConv2d(conv8, 512, (4, 4), out_size=(2, 2), strides=(2, 2),
                                    padding=pad, act=None, batch_size=batch_size, W_init=w_init, b_init=b_init, name='deconv7')
        up7 = BatchNormLayer(up7, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='dbn7')

        # print(up6.outputs)
        up6 = ConcatLayer([up7, conv7], concat_dim=3, name='concat6')
        up6 = DeConv2d(up6, 1024, (4, 4), out_size=(4, 4), strides=(2, 2),
                                    padding=pad, act=None, batch_size=batch_size, W_init=w_init, b_init=b_init, name='deconv6')
        up6 = BatchNormLayer(up6, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='dbn6')
        # print(up6.outputs)
        # exit()

        up5 = ConcatLayer([up6, conv6], concat_dim=3, name='concat5')
        up5 = DeConv2d(up5, 1024, (4, 4), out_size=(8, 8), strides=(2, 2),
                                    padding=pad, act=None, batch_size=batch_size, W_init=w_init, b_init=b_init, name='deconv5')
        up5 = BatchNormLayer(up5, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='dbn5')
        # print(up5.outputs)
        # exit()

        up4 = ConcatLayer([up5, conv5] ,concat_dim=3, name='concat4')
        up4 = DeConv2d(up4, 1024, (4, 4), out_size=(15, 15), strides=(2, 2),
                                    padding=pad, act=None, batch_size=batch_size, W_init=w_init, b_init=b_init, name='deconv4')
        up4 = BatchNormLayer(up4, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='dbn4')

        up3 = ConcatLayer([up4, conv4] ,concat_dim=3, name='concat3')
        up3 = DeConv2d(up3, 256, (4, 4), out_size=(30, 30), strides=(2, 2),
                                    padding=pad, act=None, batch_size=batch_size, W_init=w_init, b_init=b_init, name='deconv3')
        up3 = BatchNormLayer(up3, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='dbn3')

        up2 = ConcatLayer([up3, conv3] ,concat_dim=3, name='concat2')
        up2 = DeConv2d(up2, 128, (4, 4), out_size=(60, 60), strides=(2, 2),
                                    padding=pad, act=None, batch_size=batch_size, W_init=w_init, b_init=b_init, name='deconv2')
        up2 = BatchNormLayer(up2, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='dbn2')

        up1 = ConcatLayer([up2, conv2] ,concat_dim=3, name='concat1')
        up1 = DeConv2d(up1, 64, (4, 4), out_size=(120, 120), strides=(2, 2),
                                    padding=pad, act=None, batch_size=batch_size, W_init=w_init, b_init=b_init, name='deconv1')
        up1 = BatchNormLayer(up1, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='dbn1')

        up0 = ConcatLayer([up1, conv1] ,concat_dim=3, name='concat0')
        up0 = DeConv2d(up0, 64, (4, 4), out_size=(240, 240), strides=(2, 2),
                                    padding=pad, act=None, batch_size=batch_size, W_init=w_init, b_init=b_init, name='deconv0')
        up0 = BatchNormLayer(up0, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='dbn0')
        # print(up0.outputs)
        # exit()

        out = Conv2d(up0, n_out, (1, 1), act=tf.nn.sigmoid, name='out')

        print(" * Output: %s" % out.outputs)
        # exit()

    return out

## old implementation
# def u_net_2d_64_1024_deconv(x, n_out=2):
#     from tensorlayer.layers import InputLayer, Conv2d, MaxPool2d, DeConv2d, ConcatLayer
#     nx = int(x._shape[1])
#     ny = int(x._shape[2])
#     nz = int(x._shape[3])
#     print(" * Input: size of image: %d %d %d" % (nx, ny, nz))
#
#     w_init = tf.truncated_normal_initializer(stddev=0.01)
#     b_init = tf.constant_initializer(value=0.0)
#     inputs = InputLayer(x, name='inputs')
#
#     conv1 = Conv2d(inputs, 64, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv1_1')
#     conv1 = Conv2d(conv1, 64, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv1_2')
#     pool1 = MaxPool2d(conv1, (2, 2), padding='SAME', name='pool1')
#
#     conv2 = Conv2d(pool1, 128, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv2_1')
#     conv2 = Conv2d(conv2, 128, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv2_2')
#     pool2 = MaxPool2d(conv2, (2, 2), padding='SAME', name='pool2')
#
#     conv3 = Conv2d(pool2, 256, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv3_1')
#     conv3 = Conv2d(conv3, 256, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv3_2')
#     pool3 = MaxPool2d(conv3, (2, 2), padding='SAME', name='pool3')
#
#     conv4 = Conv2d(pool3, 512, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv4_1')
#     conv4 = Conv2d(conv4, 512, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv4_2')
#     pool4 = MaxPool2d(conv4, (2, 2), padding='SAME', name='pool4')
#
#     conv5 = Conv2d(pool4, 1024, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv5_1')
#     conv5 = Conv2d(conv5, 1024, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv5_2')
#
#     print(" * After conv: %s" % conv5.outputs)
#
#     up4 = DeConv2d(conv5, 512, (3, 3), out_size = (nx/8, ny/8), strides = (2, 2),
#                                 padding = 'SAME', act=None, W_init=w_init, b_init=b_init, name='deconv4')
#     up4 = ConcatLayer([up4, conv4], concat_dim=3, name='concat4')
#     conv4 = Conv2d(up4, 512, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='uconv4_1')
#     conv4 = Conv2d(conv4, 512, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='uconv4_2')
#
#     up3 = DeConv2d(conv4, 256, (3, 3), out_size = (nx/4, ny/4), strides = (2, 2),
#                                 padding = 'SAME', act=None, W_init=w_init, b_init=b_init, name='deconv3')
#     up3 = ConcatLayer([up3, conv3], concat_dim=3, name='concat3')
#     conv3 = Conv2d(up3, 256, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='uconv3_1')
#     conv3 = Conv2d(conv3, 256, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='uconv3_2')
#
#     up2 = DeConv2d(conv3, 128, (3, 3), out_size = (nx/2, ny/2), strides = (2, 2),
#                                 padding = 'SAME', act=None, W_init=w_init, b_init=b_init, name='deconv2')
#     up2 = ConcatLayer([up2, conv2] ,concat_dim=3, name='concat2')
#     conv2 = Conv2d(up2, 128, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='uconv2_1')
#     conv2 = Conv2d(conv2, 128, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='uconv2_2')
#
#     up1 = DeConv2d(conv2, 64, (3, 3), out_size = (nx/1, ny/1), strides = (2, 2),
#                                 padding = 'SAME', act=None, W_init=w_init, b_init=b_init, name='deconv1')
#     up1 = ConcatLayer([up1, conv1] ,concat_dim=3, name='concat1')
#     conv1 = Conv2d(up1, 64, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='uconv1_1')
#     conv1 = Conv2d(conv1, 64, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='uconv1_2')
#
#     conv1 = Conv2d(conv1, n_out, (1, 1), act=None, name='uconv1')
#     print(" * Output: %s" % conv1.outputs)
#     outputs = tl.act.pixel_wise_softmax(conv1.outputs)
#     return conv1, outputs
#
#
# def u_net_2d_32_1024_upsam(x, n_out=2):
#     """
#     https://github.com/jocicmarko/ultrasound-nerve-segmentation
#     """
#     from tensorlayer.layers import InputLayer, Conv2d, MaxPool2d, DeConv2d, ConcatLayer
#     batch_size = int(x._shape[0])
#     nx = int(x._shape[1])
#     ny = int(x._shape[2])
#     nz = int(x._shape[3])
#     print(" * Input: size of image: %d %d %d" % (nx, ny, nz))
#     ## define initializer
#     w_init = tf.truncated_normal_initializer(stddev=0.01)
#     b_init = tf.constant_initializer(value=0.0)
#     inputs = InputLayer(x, name='inputs')
#
#     conv1 = Conv2d(inputs, 32, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv1_1')
#     conv1 = Conv2d(conv1, 32, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv1_2')
#     pool1 = MaxPool2d(conv1, (2, 2), padding='SAME', name='pool1')
#
#     conv2 = Conv2d(pool1, 64, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv2_1')
#     conv2 = Conv2d(conv2, 64, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv2_2')
#     pool2 = MaxPool2d(conv2, (2,2), padding='SAME', name='pool2')
#
#     conv3 = Conv2d(pool2, 128, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv3_1')
#     conv3 = Conv2d(conv3, 128, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv3_2')
#     pool3 = MaxPool2d(conv3, (2, 2), padding='SAME', name='pool3')
#
#     conv4 = Conv2d(pool3, 256, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv4_1')
#     conv4 = Conv2d(conv4, 256, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv4_2')
#     pool4 = MaxPool2d(conv4, (2, 2), padding='SAME', name='pool4')
#
#     conv5 = Conv2d(pool4, 512, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv5_1')
#     conv5 = Conv2d(conv5, 512, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv5_2')
#     pool5 = MaxPool2d(conv5, (2, 2), padding='SAME', name='pool6')
#
#     # hao add
#     conv6 = Conv2d(pool5, 1024, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv6_1')
#     conv6 = Conv2d(conv6, 1024, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv6_2')
#
#     print(" * After conv: %s" % conv6.outputs)
#
#     # hao add
#     up7 = UpSampling2dLayer(conv6, (15, 15), is_scale=False, method=1, name='up7')
#     up7 =  ConcatLayer([up7, conv5], concat_dim=3, name='concat7')
#     conv7 = Conv2d(up7, 512, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv7_1')
#     conv7 = Conv2d(conv7, 512, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv7_2')
#
#     # print(nx/8,ny/8) # 30 30
#     up8 = UpSampling2dLayer(conv7, (2, 2), method=1, name='up8')
#     up8 = ConcatLayer([up8, conv4], concat_dim=3, name='concat8')
#     conv8 = Conv2d(up8, 256, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv8_1')
#     conv8 = Conv2d(conv8, 256, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv8_2')
#
#     up9 = UpSampling2dLayer(conv8, (2, 2), method=1, name='up9')
#     up9 = ConcatLayer([up9, conv3] ,concat_dim=3, name='concat9')
#     conv9 = Conv2d(up9, 128, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv9_1')
#     conv9 = Conv2d(conv9, 128, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv9_2')
#
#     up10 = UpSampling2dLayer(conv9, (2, 2), method=1, name='up10')
#     up10 = ConcatLayer([up10, conv2] ,concat_dim=3, name='concat10')
#     conv10 = Conv2d(up10, 64, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv10_1')
#     conv10 = Conv2d(conv10, 64, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv10_2')
#
#     up11 = UpSampling2dLayer(conv10, (2, 2), method=1, name='up11')
#     up11 = ConcatLayer([up11, conv1] ,concat_dim=3, name='concat11')
#     conv11 = Conv2d(up11, 32, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv11_1')
#     conv11 = Conv2d(conv11, 32, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv11_2')
#
#     conv12 = Conv2d(conv11, n_out, (1, 1), act=None, name='conv12')
#     print(" * Output: %s" % conv12.outputs)
#     outputs = tl.act.pixel_wise_softmax(conv12.outputs)
#     return conv10, outputs
#
#
# def u_net_2d_32_512_upsam(x, n_out=2):
#     """
#     https://github.com/jocicmarko/ultrasound-nerve-segmentation
#     """
#     from tensorlayer.layers import InputLayer, Conv2d, MaxPool2d, DeConv2d, ConcatLayer
#     batch_size = int(x._shape[0])
#     nx = int(x._shape[1])
#     ny = int(x._shape[2])
#     nz = int(x._shape[3])
#     print(" * Input: size of image: %d %d %d" % (nx, ny, nz))
#     ## define initializer
#     w_init = tf.truncated_normal_initializer(stddev=0.01)
#     b_init = tf.constant_initializer(value=0.0)
#     inputs = InputLayer(x, name='inputs')
#     # inputs = Input((1, img_rows, img_cols))
#     conv1 = Conv2d(inputs, 32, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv1_1')
#     # print(conv1.outputs) # (10, 240, 240, 32)
#     # conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(inputs)
#     conv1 = Conv2d(conv1, 32, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv1_2')
#     # print(conv1.outputs)    # (10, 240, 240, 32)
#     # conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv1)
#     pool1 = MaxPool2d(conv1, (2, 2), padding='SAME', name='pool1')
#     # pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
#     # print(pool1.outputs)    # (10, 120, 120, 32)
#     # exit()
#     conv2 = Conv2d(pool1, 64, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv2_1')
#     # conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool1)
#     conv2 = Conv2d(conv2, 64, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv2_2')
#     # conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv2)
#     pool2 = MaxPool2d(conv2, (2,2), padding='SAME', name='pool2')
#     # pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
#
#     conv3 = Conv2d(pool2, 128, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv3_1')
#     # conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool2)
#     conv3 = Conv2d(conv3, 128, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv3_2')
#     # conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv3)
#     pool3 = MaxPool2d(conv3, (2, 2), padding='SAME', name='pool3')
#     # pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
#     # print(pool3.outputs)   # (10, 30, 30, 64)
#
#     conv4 = Conv2d(pool3, 256, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv4_1')
#     # print(conv4.outputs)    # (10, 30, 30, 256)
#     # conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(pool3)
#     conv4 = Conv2d(conv4, 256, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv4_2')
#     # print(conv4.outputs)    # (10, 30, 30, 256) != (10, 30, 30, 512)
#     # conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv4)
#     pool4 = MaxPool2d(conv4, (2, 2), padding='SAME', name='pool4')
#     # pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
#
#     conv5 = Conv2d(pool4, 512, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv5_1')
#     # conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(pool4)
#     conv5 = Conv2d(conv5, 512, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv5_2')
#     # conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv5)
#     # print(conv5.outputs)    # (10, 15, 15, 512)
#     print(" * After conv: %s" % conv5.outputs)
#     # print(nx/8,ny/8) # 30 30
#     up6 = UpSampling2dLayer(conv5, (2, 2), name='up6')
#     # print(up6.outputs)  # (10, 30, 30, 512) == (10, 30, 30, 512)
#     up6 = ConcatLayer([up6, conv4], concat_dim=3, name='concat6')
#     # print(up6.outputs)  # (10, 30, 30, 768)
#     # up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
#     conv6 = Conv2d(up6, 256, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv6_1')
#     # conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(up6)
#     conv6 = Conv2d(conv6, 256, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv6_2')
#     # conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv6)
#
#     up7 = UpSampling2dLayer(conv6, (2, 2), name='up7')
#     up7 = ConcatLayer([up7, conv3] ,concat_dim=3, name='concat7')
#     # up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
#     conv7 = Conv2d(up7, 128, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv7_1')
#     # conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(up7)
#     conv7 = Conv2d(conv7, 128, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv7_2')
#     # conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv7)
#
#     up8 = UpSampling2dLayer(conv7, (2, 2), name='up8')
#     up8 = ConcatLayer([up8, conv2] ,concat_dim=3, name='concat8')
#     # up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
#     conv8 = Conv2d(up8, 64, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv8_1')
#     # conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up8)
#     conv8 = Conv2d(conv8, 64, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv8_2')
#     # conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv8)
#
#     up9 = UpSampling2dLayer(conv8, (2, 2), name='up9')
#     up9 = ConcatLayer([up9, conv1] ,concat_dim=3, name='concat9')
#     # up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
#     conv9 = Conv2d(up9, 32, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv9_1')
#     # conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up9)
#     conv9 = Conv2d(conv9, 32, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv9_2')
#     # conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv9)
#
#     conv10 = Conv2d(conv9, n_out, (1, 1), act=None, name='conv9')
#     # conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)
#     print(" * Output: %s" % conv10.outputs)
#     outputs = tl.act.pixel_wise_softmax(conv10.outputs)
#     return conv10, outputs


if __name__ == "__main__":
    pass
    # main()



















#
