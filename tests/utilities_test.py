import numpy as np
import pandas as pd
from hercules.utilities import interpolate_df


def test_upsampling():
    """
    Test upsampling with interpolate_df function.

    Creates a simple DataFrame with linear values and tests interpolation
    by upsampling (adding more points between existing ones).
    """
    # Create a simple dataframe with time points 0, 2, 4, 6, 8, 10
    # and linear values for 'value' column
    df = pd.DataFrame(
        {
            "time": [0, 2, 4, 6, 8, 10],
            "value": [0, 2, 4, 6, 8, 10],  # Linear function y = x
        }
    )

    # Create new_time with more points (upsampling)
    new_time = np.linspace(0, 10, 11)  # [0, 1, 2, 3, ..., 10]

    # Interpolate
    result = interpolate_df(df, new_time)

    # Assert time is correct
    assert np.allclose(result["time"], new_time)

    # Assert values are correct
    expected_values = new_time  # Linear function y = x
    assert np.allclose(result["value"], expected_values), "Interpolated values should match y = x"


def test_downsampling():
    """
    Test downsampling with interpolate_df function.

    Creates a simple DataFrame with a non-linear (quadratic) function
    and tests interpolation by downsampling (using fewer points).
    """

    time_points = np.linspace(0, 10, 11)
    df = pd.DataFrame({"time": time_points, "value": time_points * 1.7})

    # Create new_time with fewer points (downsampling)
    new_time = np.array([0, 2, 4])

    # Interpolate
    result = interpolate_df(df, new_time)

    # For our quadratic function, the interpolated values should be the square of new_time
    expected_values = new_time * 1.7
    assert np.allclose(result["value"], expected_values)

    # Check the shape is correct
    assert result.shape[0] == len(new_time)
