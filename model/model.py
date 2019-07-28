import os
import datetime
import keras
import multiprocessing
from keras import layers
import numpy as np
import keras.layers as KL
import tensorflow as tf
from data.image import normalize, resize_image
from data.utils import compose_image_meta
from data.data import data_generator
from loss.loss import neg_loss, regr_loss, ae_loss
from .module import hourglass_module, base_module, cascade_br_pool, cascade_tl_pool, center_pool
from .module import heat_layer, res_layer0
from .parallel_model import ParallelModel
import keras.models as KM
from .test_fuc import heatmap_detections, remove_by_center

def log(text, array=None):
    """Prints a text message. And, optionally, if a Numpy array is provided it
    prints it's shape, min, and max values.
    """
    if array is not None:
        text = text.ljust(25)
        text += ("shape: {:20}  ".format(str(array.shape)))
        if array.size:
            text += ("min: {:10.5f}  max: {:10.5f}".format(array.min(),array.max()))
        else:
            text += ("min: {:10}  max: {:10}".format("",""))
        text += "  {}".format(array.dtype)
    print(text)


def hourglass50(inputs, outchans, fouts):
    x = base_module(inputs, outchans[0])
    out = hourglass_module(x, outchans, fouts)
    return out


def hourglass104(inputs, outchans, fouts):
    x = base_module(inputs, outchans[0])
    x = hourglass_module(x, outchans, fouts)
    out = hourglass_module(x, outchans, fouts)
    return out


# In author's implementation, for hourglass-104, he repeat the head twice sequent.we not
def head(inputs,fouts,nbclass):
    out = []
    x_tl = cascade_tl_pool(inputs, fouts)
    x_br = cascade_br_pool(inputs, fouts)
    x_center = center_pool(inputs, fouts)

    # for heatmap bias, set it as -2.19
    tl_heatmap = heat_layer(x_tl, fouts, nbclass)
    tl_heatmap = layers.Activation('sigmoid')(tl_heatmap)
    out.append(tl_heatmap)
    br_heatmap = heat_layer(x_br, fouts, nbclass)
    br_heatmap = layers.Activation('sigmoid')(br_heatmap)
    out.append(br_heatmap)
    center_heatmap = heat_layer(x_center, fouts, nbclass)
    center_heatmap = layers.Activation('sigmoid')(center_heatmap)
    out.append(center_heatmap)
    tl_tag = heat_layer(x_tl, fouts, 1)
    tl_tag = layers.Flatten()(tl_tag)
    out.append(tl_tag)
    br_tag = heat_layer(x_br, fouts, 1)
    br_tag = layers.Flatten()(br_tag)
    out.append(br_tag)

    tl_reg = heat_layer(x_tl, fouts, 2)
    tl_reg = layers.Activation('sigmoid')(tl_reg)
    out.append(tl_reg)
    br_reg = heat_layer(x_br, fouts, 2)
    br_reg = layers.Activation('sigmoid')(br_reg)
    out.append(br_reg)
    cent_reg = heat_layer(x_center, fouts, 2)
    cent_reg = layers.Activation('sigmoid')(cent_reg)
    out.append(cent_reg)

    return out


class CenterNet52():

    def __init__(self, mode, config, model_dir):
        """
        mode: Either "training" or "inference"
        config: A Sub-class of the Config class
        model_dir: Directory to save training logs and trained weights
        """
        assert mode in ['training', 'inference']
        self.mode = mode
        self.config = config
        self.model_dir = model_dir
        self.set_log_dir()
        self.keras_model = self.build(mode=mode)

    def build(self, mode):
        """Build Mask R-CNN architecture.
            input_shape: The shape of the input image.
            mode: Either "training" or "inference". The inputs and
                outputs of the model differ accordingly.
        """
        assert mode in ['training', 'inference']

        # Inputs
        input_image = KL.Input(
            shape=self.config.VIEW_SIZE+[3], name="input_image")
        input_image_meta = KL.Input(shape=[self.config.MAX_NUMS, self.config.META_SHAPE],
                                    name="input_image_meta")
        if mode == "training":
            # heatmaps and regress maps
            input_tl_heatmaps = KL.Input(
                shape = self.config.OUTPUT_SIZE+[self.config.CLASSES], name="input_tl_heatmaps", dtype=tf.float32)
            input_br_heatmaps = KL.Input(
                shape = self.config.OUTPUT_SIZE+[self.config.CLASSES], name="input_br_heatmaps", dtype=tf.float32)
            input_ct_heatmaps = KL.Input(
                shape= self.config.OUTPUT_SIZE+[self.config.CLASSES], name="input_ct_heatmaps", dtype=tf.float32)

            input_tl_reg = KL.Input(
                shape= self.config.OUTPUT_SIZE+[2], name="input_tl_reg", dtype=tf.float32)
            input_br_reg = KL.Input(
                shape= self.config.OUTPUT_SIZE+[2], name="input_br_reg", dtype=tf.float32)
            input_ct_reg = KL.Input(
                shape= self.config.OUTPUT_SIZE+[2], name="input_ct_reg", dtype=tf.float32)

            input_mask = KL.Input(
                shape=[3] + self.config.OUTPUT_SIZE, name="input_mask", dtype=tf.float32)
            input_tag_mask = KL.Input(
                shape=[self.config.MAX_NUMS], name="input_tag_mask", dtype=tf.float32)
            input_tl_tag = KL.Input(
                shape=[self.config.MAX_NUMS], name="input_tl_tag", dtype=tf.int64)
            input_br_tag = KL.Input(
                shape=[self.config.MAX_NUMS], name="input_br_tag", dtype=tf.int64)
            input_gt_bbox = KL.Input(
                shape=[self.config.MAX_NUMS, 4], name="input_gt_bbox", dtype=tf.float32)
            input_gt_class_id = KL.Input(
                shape=[self.config.MAX_NUMS], name="input_gt_class_id", dtype=tf.int64)

        # Build the center network graph
        x = base_module(input_image, self.config.INTER_CHANNELS[0])
        backbone_feat = hourglass_module(x, self.config.INTER_CHANNELS, self.config.NUM_FEATS)
        outs = head(backbone_feat, self.config.NUM_FEATS, self.config.CLASSES)

        if mode == "training":
            # Losses
            # heatmaps loss
            tl_map_loss = KL.Lambda(lambda x: neg_loss(*x), name="tl_map_loss")([outs[0], input_tl_heatmaps])
            br_map_loss = KL.Lambda(lambda x: neg_loss(*x), name="br_map_loss")([outs[1], input_br_heatmaps])
            ct_map_loss = KL.Lambda(lambda x: neg_loss(*x), name="ct_map_loss")([outs[2], input_ct_heatmaps])

            # regression loss
            tl_mask, br_mask, ct_mask = KL.Lambda(lambda x :tf.unstack(x, axis=1), name="unstack_mask")(input_mask)
            lt_reg_loss = KL.Lambda(lambda x: regr_loss(*x), name="tl_reg_loss")([outs[5], input_tl_reg, tl_mask])
            br_reg_loss = KL.Lambda(lambda x: regr_loss(*x), name="br_reg_loss")([outs[6], input_br_reg, br_mask])
            ct_reg_loss = KL.Lambda(lambda x: regr_loss(*x), name="ct_reg_loss")([outs[7], input_ct_reg, ct_mask])

            # embedding loss
            pull_push_loss = KL.Lambda(lambda x: ae_loss(*x),
                                       name="ae_loss")([outs[3], outs[4], input_tl_tag, input_br_tag, input_tag_mask])

            # Model
            inputs = [input_image, input_image_meta, input_tl_heatmaps, input_br_heatmaps,
                      input_ct_heatmaps, input_tl_reg, input_br_reg, input_ct_reg, input_mask, input_tl_tag,
                      input_br_tag, input_tag_mask, input_gt_bbox, input_gt_class_id]

            outputs = outs + [tl_map_loss, br_map_loss, ct_map_loss, lt_reg_loss, br_reg_loss,
                              ct_reg_loss, pull_push_loss]
            model = KM.Model(inputs, outputs, name='centernet52')
        else:

            model = KM.Model([input_image, input_image_meta],
                             outs,
                             name='centernet52')
        if self.config.NUM_GPUS > 1:
            model = ParallelModel(model, self.config.NUM_GPUS)
        return model

    def set_log_dir(self, model_path=None):
        """Sets the model log directory and epoch counter.
        model_path: If None, or a format different from what this code uses
            then set a new log directory and start epochs from 0. Otherwise,
            extract the log directory and the epoch counter from the file
            name.
        """
        # Set date and epoch counter as if starting a new model
        self.epoch = 0
        # Directory for training logs
        self.log_dir = os.path.join(self.model_dir,self.config.NET_NAME.lower())

        # Path to save after each epoch. Include placeholders that get filled by Keras.
        self.checkpoint_path = os.path.join(self.log_dir, "centernet52_{epoch:02d}.h5")

    def find_last(self):
        """Finds the last checkpoint file of the last trained model in the
        model directory.
        Returns:
            The path of the last checkpoint file
        """
        # Get directory names. Each directory corresponds to a model
        dir_names = next(os.walk(self.model_dir))[1]
        key = self.config.NET_NAME.lower()
        dir_names = filter(lambda f: f.startswith(key), dir_names)
        dir_names = sorted(dir_names)
        if not dir_names:
            import errno
            raise FileNotFoundError(
                errno.ENOENT,
                "Could not find model directory under {}".format(self.model_dir))
        # Pick last directory
        dir_name = os.path.join(self.model_dir, dir_names[-1])
        # Find the last checkpoint
        checkpoints = next(os.walk(dir_name))[2]
        checkpoints = filter(lambda f: f.startswith("centernet52"), checkpoints)
        checkpoints = sorted(checkpoints)
        if not checkpoints:
            import errno
            raise FileNotFoundError(
                errno.ENOENT, "Could not find weight files in {}".format(dir_name))
        checkpoint = os.path.join(dir_name, checkpoints[-1])
        return checkpoint

    def load_weights(self, filepath, by_name=False, exclude=None):

        import h5py
        try:
            from keras.engine import saving
        except ImportError:
            # Keras before 2.2 used the 'topology' namespace.
            from keras.engine import topology as saving

        if exclude:
            by_name = True

        if h5py is None:
            raise ImportError('`load_weights` requires h5py.')
        f = h5py.File(filepath, mode='r')
        if 'layer_names' not in f.attrs and 'model_weights' in f:
            f = f['model_weights']

        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        keras_model = self.keras_model
        layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model") \
            else keras_model.layers

        # Exclude some layers
        if exclude:
            layers = filter(lambda l: l.name not in exclude, layers)

        if by_name:
            saving.load_weights_from_hdf5_group_by_name(f, layers)
        else:
            saving.load_weights_from_hdf5_group(f, layers)
        if hasattr(f, 'close'):
            f.close()

        # Update the log directory
        self.set_log_dir(filepath)

    def compile(self, learning_rate, momentum):
        """Gets the model ready for training. Adds losses, regularization, and
        metrics. Then calls the Keras compile() function.
        """
        # Optimizer object
        optimizer = keras.optimizers.SGD(
            lr=learning_rate, momentum=momentum,
            clipnorm=self.config.GRADIENT_CLIP_NORM)
        # Add Losses
        # First, clear previously set losses to avoid duplication
        self.keras_model._losses = []
        self.keras_model._per_input_losses = {}
        loss_names = ["tl_map_loss",  "br_map_loss", "ct_map_loss",
                      "tl_reg_loss", "br_reg_loss", "ct_reg_loss", "ae_loss"]
        for name in loss_names:
            layer = self.keras_model.get_layer(name)
            if layer.output in self.keras_model.losses:
                continue
            loss = (
                tf.reduce_mean(layer.output, keepdims=True)
                * self.config.LOSS_WEIGHTS.get(name, 1.))
            self.keras_model.add_loss(loss)

        # Add L2 Regularization
        # Skip gamma and beta weights of batch normalization layers.
        reg_losses = [
            keras.regularizers.l2(self.config.WEIGHT_DECAY)(w) / tf.cast(tf.size(w), tf.float32)
            for w in self.keras_model.trainable_weights
            if 'gamma' not in w.name and 'beta' not in w.name]
        self.keras_model.add_loss(tf.add_n(reg_losses))

        # Compile
        self.keras_model.compile(
            optimizer=optimizer,
            loss=[None] * len(self.keras_model.outputs))

        # Add metrics for losses
        for name in loss_names:
            if name in self.keras_model.metrics_names:
                continue
            layer = self.keras_model.get_layer(name)
            self.keras_model.metrics_names.append(name)
            loss = (layer.output * self.config.LOSS_WEIGHTS.get(name, 1.))
            self.keras_model.metrics_tensors.append(loss)

    def train(self, train_dataset, val_dataset, learning_rate, epochs,
              augment=None, custom_callbacks=None):
        """Train the model.
        train_dataset, val_dataset: Training and validation Dataset objects.
        learning_rate: The learning rate to train with
        epochs: Number of training epochs.
        augment: Optional.
        """
        assert self.mode == "training", "Create model in training mode."

        # Data generators
        train_generator = data_generator(train_dataset, self.config, shuffle=True,
                                         augment=augment)
        val_generator = data_generator(val_dataset, self.config)

        # Create log_dir if it does not exist
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # Callbacks
        callbacks = [
            keras.callbacks.TensorBoard(log_dir=self.log_dir,
                                        histogram_freq=0, write_graph=True, write_images=False),
            keras.callbacks.ModelCheckpoint(self.checkpoint_path,
                                            verbose=0, save_weights_only=True),
        ]

        # Add custom callbacks to the list
        if custom_callbacks:
            callbacks += custom_callbacks

        # Train
        log("\nStarting at epoch {}. LR={}\n".format(self.epoch, learning_rate))
        log("Checkpoint Path: {}".format(self.checkpoint_path))
        self.compile(learning_rate, self.config.MOMENTUM)

        # Work-around for Windows: Keras fails on Windows when using
        # multiprocessing workers. See discussion here:
        # https://github.com/matterport/Mask_RCNN/issues/13#issuecomment-353124009
        if os.name is 'nt':
            workers = 0
        else:
            workers = multiprocessing.cpu_count()

        self.keras_model.fit_generator(
            train_generator,
            initial_epoch=self.epoch,
            epochs=epochs,
            steps_per_epoch=self.config.STEPS_PER_EPOCH,
            callbacks=callbacks,
            validation_data=val_generator,
            validation_steps=self.config.VAL_STEPS,
            max_queue_size=100,
            workers=workers,
            use_multiprocessing=True,
        )
        self.epoch = max(self.epoch, epochs)

    def mold_inputs(self, images):
        """Takes a list of images and modifies them to the format expected
        as an input to the neural network.
        images: List of image matrices [height,width,depth]. Images can have
            different sizes.
        Returns 3 Numpy matrices:
        molded_images: [N, h, w, 3]. Images resized and normalized.
        image_metas: [N, length of meta data]. Details about each image.
        windows: [N, (y1, x1, y2, x2)]. The portion of the image that has the
            original image (padding excluded).
        """
        molded_images = []
        image_metas = []
        windows = []
        for image in images:
            # Resize image
            # TODO: move resizing to mold_image()
            original_shape = image.shape
            normalize(image, self.config.MEANS, self.config.STD)
            image, scale, window = resize_image(image, self.config.VIEW_SIZE)
            # Build image_meta
            image_meta = compose_image_meta(
                0, original_shape, image.shape, window, scale,
                np.zeros([self.config.NUM_CLASSES], dtype=np.int32))
            # Append
            molded_images.append(image)
            windows.append(window)
            image_metas.append(image_meta)
        # Pack into arrays
        molded_images = np.stack(molded_images)
        image_metas = np.stack(image_metas)
        windows = np.stack(windows)
        return molded_images, image_metas, windows

    def unmold_detections(self, detections, original_image_shape,
                          image_shape, window):
        """Reformats the detections of one image from the format of the neural
        network output to a format suitable for use in the rest of the
        application.
        detections: [N, (x1, y1, x2, y2, score, class_id)] i
        original_image_shape: [H, W, C] Original image shape before resizing
        image_shape: [H, W, C] Shape of the image after resizing and padding
        window: [y1, x1, h, w] Pixel coordinates of box in the image where the real
                image is excluding the padding.
        Returns:
        boxes: [N, (x1, y1, x2, y2)] Bounding boxes in pixels
        class_ids: [N] Integer class IDs for each bounding box
        scores: [N] Float probability scores of the class_id
        """
        # How many detections do we have?
        # Detections array is padded with zeros. Find the first class_id == 0.
        zero_ix = np.where(detections[:, 5] == 0)[0]
        N = zero_ix[0] if zero_ix.shape[0] > 0 else detections.shape[0]

        # Extract boxes, class_ids, scores, and class-specific masks
        boxes = detections[:N, :4]
        scores = detections[:N, 4]
        class_ids = detections[:N, 5].astype(np.int32)

        # Translate normalized coordinates in the resized image to pixel
        # coordinates in the original image before resizing
        wy1, wx1, wh, ww = window
        shift = np.array([wy1, wx1, wy1, wx1])
        scale = np.min(image_shape/original_image_shape)
        # Convert boxes to normalized coordinates on the window
        boxes = np.divide(boxes - shift, scale)
        # Filter out detections with zero area. Happens in early training when
        # network weights are still random
        exclude_ix = np.where(
            (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) <= 0)[0]
        if exclude_ix.shape[0] > 0:
            boxes = np.delete(boxes, exclude_ix, axis=0)
            class_ids = np.delete(class_ids, exclude_ix, axis=0)
            scores = np.delete(scores, exclude_ix, axis=0)

        return boxes, class_ids, scores

    def detect(self, images, verbose=0):
        """Runs the detection pipeline.
        images: List of images, potentially of different sizes.
        Returns a list of dicts, one dict per image. The dict contains:
        rois: [N, (y1, x1, y2, x2)] detection bounding boxes
        class_ids: [N] int class IDs
        scores: [N] float probability scores for the class IDs
        """
        assert self.mode == "inference", "Create model in inference mode."
        assert len(
            images) == self.config.BATCH_SIZE, "len(images) must be equal to BATCH_SIZE"

        if verbose:
            log("Processing {} images".format(len(images)))
            for image in images:
                log("image", image)

        # Mold inputs to format expected by the neural network
        molded_images, image_metas, windows = self.mold_inputs(images)

        # Validate image sizes
        # All images in a batch MUST be of the same size
        image_shape = molded_images[0].shape
        for g in molded_images[1:]:
            assert g.shape == image_shape, \
                "After resizing, all images must have the same size. Check IMAGE_RESIZE_MODE and image sizes."
        if verbose:
            log("molded_images", molded_images)
            log("image_metas", image_metas)
        # Run object detection
        out = self.keras_model.predict([molded_images, image_metas], verbose=0)
        detections, centers = heatmap_detections(out, self.config)
        final_detections = remove_by_center(detections, centers, self.config)
        # Process detections
        results = []
        for i, image in enumerate(images):
            final_rois, final_class_ids, final_scores = \
                self.unmold_detections(final_detections[i], image.shape, molded_images[i].shape,
                                       windows[i])
            results.append({
                "rois": final_rois,
                "class_ids": final_class_ids,
                "scores": final_scores,
            })
        return results












