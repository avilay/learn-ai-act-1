from configparser import ConfigParser


class Config:
    def __init__(self, **kwargs):
        self.tblogs = kwargs.get("tblogs")

    def __repr__(self):
        return f"<Config(tblogs={self.tblogs})>"

    def load(self, conffile):
        conf = ConfigParser()
        with open(conffile, "rt") as f:
            conf.read(f)
        self.tblogs = conf["DEFAULT"].get("tblogs")


config = Config()
