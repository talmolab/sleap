from sleap.gui.activities.inference.model import InferenceGuiModel


class InferenceGuiController(object):
    model: InferenceGuiModel

    def run(self):
        print("+++ Run stub...")
        print(
            "sleap-track C:/Users/ariem/work/sleap_data/videos/centered_pair_small.mp4 "
            "--frames 225,611,996,675,199,520,9,409,523,75,1067,591,656,401,271,372,821,694,345,477 "
            "-m C:/Users/ariem/work/sleap_data\models\210316_114545.centroid.n=4\training_config.json "
            "-m C:/Users/ariem/work/sleap_data\models\210316_114734.centered_instance.n=4\training_config.json "
            "--tracking.tracker none "
            "-o C:/Users/ariem/work/sleap_data\predictions\centered_pair_small.mp4.210405_103129.predictions.slp "
            "--verbosity json --no-empty-frames"
        )

    def save(self):
        print("+++ Save stub...")

    def export(self):
        print("+++ Export stub...")

    def load(self):
        print("+++ Load stub...")
