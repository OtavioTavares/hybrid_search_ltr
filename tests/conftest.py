import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def products():
    return pd.DataFrame(
        {
            "product_name": ["gold leg chair", "wood table", "blue sofa"],
            "product_class": ["Chairs", "Tables", "Sofas"],
            "category_hierarchy": ["Furniture / Chairs", "Furniture / Tables", "Furniture / Sofas"],
            "product_description": ["", "nice", "soft"],
            "product_features": ["legcolor : gold|material : velvet", "", "width : 24.75|color : blue"],
            "attributes": ["gold velvet", "", "24.75 blue"],
            "rating_count": [10, np.nan, 3],
            "average_rating": [4.5, np.nan, 3.0],
            "review_count": [5, np.nan, 1],
        }
    )
