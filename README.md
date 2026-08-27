Immune Network Simulator

Author: lopzedu-arch


This code is part of the work presented in the article: Macrophage–lymphocyte interaction in the
immune response to SARS-CoV-2: a dynamic functional model. It allows the simulation of the immune network dynamics proposed within the aforementioned article.

The code is structured into three fundamental components:

    Solving of differential equations – Numerical integration of the network dynamics.

    Transformation of boolean expressions – Conversion into their fuzzy-logic probabilistic interpretation.

    Interactive interface – A graphical user interface for running and visualizing simulations.

Compatibility and Execution

There are no restrictions regarding the type of notebook or environment that can be used. However, we recommend the following options:
Option 1: Google Colab (Quick setup, cloud-based)

    Click on  MacroLymph_Net_July28.ipynb in this repository.

    Select the option open in Collab.

Option 2: Local Environment (Recommended for speed)

Running the code on your local machine significantly improves performance:
    
    Copy the from the aforementioned google Collab page and paste it your notebook.
    
    Expected speed: ~2 seconds per simulation (compared to ~10 seconds on Colab).

    Requirements: Python 3.8+ and the dependencies listed in requirements.txt.
