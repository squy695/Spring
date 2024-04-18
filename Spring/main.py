from others.MicroScaler import MicroScaler
from others.Showar import Showar
from others.Spring import Spring
from config.Config import Config


def initController(name: str, config: Config):
    if name == "MicroScaler":
        return MicroScaler(config)
    elif name == "SHOWAR":
        return Showar(config)
    elif name == "Spring":
        return Spring(config)
    else:
        raise NotImplementedError()


if __name__ == "__main__":
    config = Config()

    controller = initController("Spring", config)
    controller.start()
