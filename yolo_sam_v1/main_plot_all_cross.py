from constants import ALL_MODELS
from functions import ModelPlotter, ResultPlotter

if __name__ == "__main__":

    for MODEL in ALL_MODELS:
        # === LOAD MODEL ===
        plotter = ModelPlotter(MODEL, mode="cross")
        plotter.load_model()

        # === IMAGES ===
        plotter.prepare_images()
        plotter.plot()

    # Create comparison plots
    plotter = ResultPlotter(ALL_MODELS, mode="cross")
    plotter.create_all_models_comparison(image_index=7)