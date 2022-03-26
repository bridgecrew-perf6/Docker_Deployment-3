{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "instantiate_model.ipynb",
      "provenance": [],
      "collapsed_sections": [],
      "authorship_tag": "ABX9TyObyslIYNbsvg0xSVl8F7Of",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/lfgranellosouza/Docker_Deployment/blob/main/instantiate_model.ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "faLPHZ3jv340"
      },
      "outputs": [],
      "source": [
        "import joblib\n",
        "\n",
        "import utils.data_processor as dp\n",
        "import utils.model_trainer as mt\n",
        "\n",
        "path_to_data = './data/spam.csv'\n",
        "\n",
        "prep_data = dp.prepare_data(path_to_data, encoding= 'latin-1')\n",
        "\n",
        "train_test_data, vectorizer = dp.create_train_test_data(prep_data['text'],\n",
        "                                                        prep_data['label'],\n",
        "                                                        test_size_pct= .2,\n",
        "                                                        random_state= 2022\n",
        "                                                        )\n",
        "\n",
        "model = mt.run_model_training(train_test_data['X_train'], train_test_data['X_test'],\n",
        "                              train_test_data['y_train'], train_test_data['y_test']\n",
        "                              )\n",
        "\n",
        "joblib.dump(model, './models/spam_detector_model.pkl')\n",
        "joblib.dump(vectorizer, open(\"./vectors/vectorizer.pickle\", 'wb'))"
      ]
    }
  ]
}
