import pytorch_lightning as pl


def _install_safe_checkpoint():
    if getattr(pl.Trainer, '_save_checkpoint_wrapped', False):
        return

    _orig_trainer_save_checkpoint = pl.Trainer.save_checkpoint

    def _safe_save_checkpoint(self, filepath, *args, **kwargs):
        try:
            return _orig_trainer_save_checkpoint(self, filepath, *args, **kwargs)
        except OSError as e:
            try:
                err_no = e.errno
            except Exception:
                err_no = None
            if err_no == 28 or (isinstance(e, OSError) and 'No space' in str(e)):
                print(f"Warning: failed to save checkpoint {filepath}: {e}. Continuing without saving.")
                return None
            raise

    pl.Trainer.save_checkpoint = _safe_save_checkpoint
    pl.Trainer._save_checkpoint_wrapped = True


_install_safe_checkpoint()
