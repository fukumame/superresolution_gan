# -*- coding: utf-8 -*-
import argparse
import glob
import logging
import os
import pickle
import chainer
import numpy as np
from chainer.links import VGG19Layers
from chainer.dataset import concat_examples

import srcgan

parser = argparse.ArgumentParser()
parser.add_argument("--src", "-s", required=True)
parser.add_argument("--gpu", "-g", type=int, default=-1)
parser.add_argument("--batchsize", "-b", type=int, default=10)
parser.add_argument("--epoch", "-e", type=int, default=10)
parser.add_argument("--outdirname", "-o", required=True)
parser.add_argument("--vgg", "-v", action="store_true")
parser.add_argument("--vgg_layers", "-vl", type=str, default="conv5_4")
parser.add_argument("--pretrained_generator")
parser.add_argument("--k_adversarial", type=float, default=1)
parser.add_argument("--k_mse", type=float, default=1)
args = parser.parse_args()

OUTPUT_DIRECTORY = args.outdirname

if not os.path.isdir(OUTPUT_DIRECTORY):
    os.makedirs(OUTPUT_DIRECTORY)

logging.basicConfig(filename=os.path.join(OUTPUT_DIRECTORY, "log.txt"), level=logging.DEBUG)
console = logging.StreamHandler()
logging.getLogger('').addHandler(console)

logging.info(args)
if args.pretrained_generator is not None:
    logging.info("pretrained_generator: {}".format(os.path.abspath(args.pretrained_generator)))


# paths = glob.glob(args.src)
dataset = srcgan.dataset.PreprocessedImageDataset(src=args.src, cropsize=96, resize=(300, 300))

iterator = chainer.iterators.MultiprocessIterator(dataset, batch_size=args.batchsize, repeat=True, shuffle=True)

generator = srcgan.models.SRGenerator()
if args.pretrained_generator is not None:
    chainer.serializers.load_npz(args.pretrained_generator, generator)
if args.gpu >= 0:
    generator.to_gpu()

discriminator = srcgan.models.SRDiscriminator()
if args.gpu >= 0:
    discriminator.to_gpu()
optimizer_discriminator = chainer.optimizers.Adam()
optimizer_discriminator.setup(discriminator)

optimizer_generator = chainer.optimizers.Adam()
optimizer_generator.setup(generator)

count_processed = 0
sum_loss_generator, sum_loss_generator_adversarial, sum_loss_generator_content = 0, 0, 0

if args.vgg:
    vgg = VGG19Layers()


while iterator.epoch < args.epoch:
    train_batch = iterator.next()
    low_res, high_res = concat_examples(train_batch, args.gpu)

    super_res = generator(low_res)

    discriminated_from_super_res = discriminator(super_res)
    discriminated_from_high_res = discriminator(high_res)
    loss_generator_adversarial = chainer.functions.softmax_cross_entropy(
        discriminated_from_super_res,
        chainer.Variable(np.zeros(discriminated_from_super_res.data.shape[0], dtype=np.int32))
    )

    if not args.vgg:
        loss_generator_content = chainer.functions.mean_squared_error(
            super_res,
            high_res
        )
    else:
        loss_generator_content = chainer.functions.mean_squared_error(
            vgg.forward(super_res, layers=[args.vgg_layers])[args.vgg_layers],
            vgg.forward(high_res, layers=[args.vgg_layers])[args.vgg_layers]
        )

    loss_generator = loss_generator_content * args.k_mse + loss_generator_adversarial * args.k_adversarial
    sum_loss_generator_adversarial += loss_generator_adversarial.data
    sum_loss_generator_content += loss_generator_content.data

    loss_discriminator = chainer.functions.softmax_cross_entropy(
        discriminated_from_super_res,
        chainer.Variable(np.ones(discriminated_from_super_res.data.shape[0], dtype=np.int32))
    ) + chainer.functions.softmax_cross_entropy(
        discriminated_from_high_res,
        chainer.Variable(np.zeros(discriminated_from_high_res.data.shape[0], dtype=np.int32))
    )

    generator.cleargrads()
    loss_generator.backward()
    optimizer_generator.update()

    discriminator.cleargrads()
    loss_discriminator.backward()
    optimizer_discriminator.update()

    sum_loss_generator += loss_generator.data

    report_span = args.batchsize * 10
    count_processed += len(super_res.data)
    if count_processed % report_span == 0:
        logging.info("processed: {}".format(count_processed))
        logging.info("loss_generator: {}".format(sum_loss_generator / report_span))
        logging.info("loss_generator_adversarial: {}".format(sum_loss_generator_adversarial / report_span))
        logging.info("loss_generator_mse: {}".format(sum_loss_generator_content / report_span))
        sum_loss_generator, sum_loss_generator_adversarial, sum_loss_generator_content = 0, 0, 0

    if iterator.is_new_epoch:
        chainer.serializers.save_npz(
            os.path.join(OUTPUT_DIRECTORY, "generator_model_epoch-{}.npz".format(iterator.epoch)), generator)
