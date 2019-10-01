"""Module that contains library callbacks."""
# from abc import ABC, abstractmethod


class Callback:
    """Callback interface."""

    def on_train_begin(self, info):
        """Called when training procedure is initiated."""
        return

    def on_train_end(self, info):
        """Called when training procedure has ended."""
        return

    def on_epoch_begin(self, info):
        """Called at the start of each epoch."""
        return

    def on_epoch_end(self, info):
        """Called at the end of each epoch."""
        return

    def on_batch_begin(self, info):
        """Called before processing each batch."""
        return

    def on_batch_end(self, info):
        """Called after processing each batch."""
        return


class TensorboardLoggerCallback(Callback):
    """Tensorboard callback that logs metrics at the end of each batch, and epoch."""

    def __init__(self, logger_destination):
        self.writer = Logger(logger_destination)
        print("> writing tensorboard to", logger_destination)

    def on_batch_end(self, info):
        global_step = info["global_step"]
        self.writer("train_arc_acc", info["arc_accuracy"], global_step)
        self.writer("train_rel_acc", info["rel_accuracy"], global_step)
        self.writer("train_arc_loss", info["arc_loss"], global_step)
        self.writer("train_rel_loss", info["rel_loss"], global_step)

    def on_epoch_end(self, info):
        self.writer("dev_uas", info["dev_uas"], info["epoch"])
        self.writer("dev_las", info["dev_las"], info["epoch"])

    def raw_write(self, variable, value):
        self.writer(variable, value, 0)


class ModelSaveCallback(Callback):
    """Callback that saves model between epochs."""

    def __init__(self, save_destination, save_after=0, mode="best"):
        self._modes = {
            "best": self._best_save,  # save only best models (as evaluated on dev)
            "epoch": self._epoch_save,  # save model on each epoch
        }

        self.save_destination = save_destination
        self.save_after = save_after
        self.best_uas = -1
        self.best_epoch = -1

        _mode_names = ", ".join(self._modes.keys())
        assert mode in self._modes, "Mode not found in [%s]" % _mode_names

        self._save_function = self._modes[mode]

        print(f"> Saving model to {save_destination} (after step {save_after})")

    def on_epoch_end(self, info):
        self._save_function(info)

    def _best_save(self, info):
        dev_uas = info["dev_uas"]
        global_step = info["global_step"]

        if (dev_uas < self.best_uas) or (global_step < self.save_after):
            # skipping
            return
        else:
            self.best_uas = dev_uas
            self.best_epoch = info["epoch"]
            info["model"].save_to_file(self.save_destination)
            print("saved to", self.save_destination)

    def _epoch_save(self, info):
        outputfile = "%s_%s" % (self.save_destination, info["epoch"])
        info["model"].save_to_file(outputfile)


class Logger:
    """Logging in tensorboard without tensorflow ops."""

    def __init__(self, log_dir):
        """Creates a summary writer logging to log_dir."""
        # import is located inside body to avoid a dependency
        # on tensorboard / tensorflow
        import tensorflow as tf

        self.writer = tf.summary.FileWriter(log_dir)
        self.metrics = {}
        self.tf = tf

    def __call__(self, tag, value, step):
        """Log a scalar variable.
        Parameter
        ----------
        tag : basestring
            Name of the scalar
        value : scalar value
            value
        step : int
            training iteration
        """
        if tag not in self.metrics:
            _metric = self.tf.Summary()
            _metric.value.add(tag=tag, simple_value=None)
            self.metrics[tag] = _metric

        summary = self.metrics[tag]

        summary.value[0].simple_value = value
        self.writer.add_summary(summary, step)
