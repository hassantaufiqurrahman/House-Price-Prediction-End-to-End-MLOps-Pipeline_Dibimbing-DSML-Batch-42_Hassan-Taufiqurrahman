# Import Library
import json
import logging
import os

from . import config

# Mengubah Satu Catatan Log menjadi Satu Baris JSON
class FormatJSON(logging.Formatter):
    def format(self, catatan):
        isi = {
            "ts": self.formatTime(catatan, "%Y-%m-%dT%H:%M:%S"),
            "level": catatan.levelname,
            "event": catatan.getMessage(),
        }
        # Field tambahan yang Dikirim Lewat extra={...}
        bawaan = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys())
        bawaan.add("message")
        bawaan.add("asctime")
        for kunci in catatan.__dict__:
            if kunci not in bawaan:
                isi[kunci] = catatan.__dict__[kunci]
        return json.dumps(isi, ensure_ascii=False)

# Membuat Logger yang Menulis ke Layar dan Ke "logs/predictions.log"
def ambil_logger(nama="rumahin"):
    log = logging.getLogger(nama)
    if len(log.handlers) > 0:
        return log                      

    log.setLevel(logging.INFO)

    os.makedirs(os.path.dirname(config.FILE_LOG), exist_ok=True)
    ke_file = logging.FileHandler(config.FILE_LOG, encoding="utf-8")
    ke_file.setFormatter(FormatJSON())
    log.addHandler(ke_file)

    ke_layar = logging.StreamHandler()
    ke_layar.setFormatter(FormatJSON())
    log.addHandler(ke_layar)

    log.propagate = False
    return log
