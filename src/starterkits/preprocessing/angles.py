# ©, 2019, Sirris
# owner: RVEB

"""Tools to convert angles in degrees or radians into triangular or trigonometric variables"""

import numpy as np


def s_func(x, kind='deg'):
    """
    Transform an angular variable to the s triangular variable.

    The triangular functions s and c are similar to sine and cosine, respectively, but they are linear functions of
    the angular variables.
    They have the property that |s| + |c| = 1

    :param x: numpy array or pandas series containing the variable in degrees or radians
    :param kind: specifying whether x is in degrees or radians. Accepts 'deg', 'degrees', 'rad', or 'radians'

    :returns: array or series of the same length as x, with the values corresponding to the s triangular function
    """
    y = np.copy(x)
    kind = kind.lower()
    assert kind in ('deg', 'degrees', 'rad', 'radians')
    if kind[:3] == 'deg':
        full_circle = 360
    else:
        full_circle = 2 * np.pi

    y = ((y + full_circle / 4) % full_circle - full_circle / 4) / (full_circle / 4)
    q = np.maximum(0, np.sign(y - 1))
    y = y - q * 2 * (y - q)

    return y


def c_func(x, kind='deg'):
    """
    Transform an angular variable to the c triangular variable.

    :param x: numpy array or pandas series containing the variable in degrees or radians
    :param kind: specifying whether x is in degrees or radians. Accepts 'deg', 'degrees', 'rad', or 'radians'

    :return: array or series of the same length as x, with the values corresponding to the c triangular function

    """
    if kind in ('deg', 'degrees'):
        shift = 90
    else:
        shift = np.pi/2
    y = s_func(x + shift, kind=kind)
    return y


def revert_s_c_to_degrees(s, c, kind='deg'):
    """
    Convert the s and c variables back to the original degrees

    :param s: s triangular variable
    :param c: c triangular variable
    :param kind: specifying whether the output should be in degrees or radians. Accepts 'deg', 'degrees', 'rad',
     or 'radians'

    :return: array or series of the same length as x, with the values corresponding to the angles in degrees or radians.

    """
    kind = kind.lower()
    assert kind in ('deg', 'degrees', 'rad', 'radians')
    if kind[:3] == 'deg':
        full_circle = 360
    else:
        full_circle = 2 * np.pi
    q = np.minimum(0, np.sign(s)) * -1
    y = ((-c + 2 * q * (q + c)) + 1) * (full_circle / 4)
    return y


def revert_sin_cos_to_degrees(sin, cos, kind='deg'):
    """
    Convert the s and c variables back to the original degrees

    :param sin: s variable
    :param cos: c variable
    :param kind: specifies whether the output should be in degrees ('deg', 'degrees') or radians ('rad', 'radians')

    :return: array or series of the same length as x, with the values corresponding to the angles in degrees or radians.

    """
    acos = np.arccos(cos)
    asin = np.arcsin(sin)
    # q is 1 in the first two quadrants (between 0 and 180 degrees), -1 in the 3rd and 4th quadrant
    q = (asin > 0).astype(np.int64) * 2 - 1
    # If the angle should be in the 3rd or 4th quadrant, mirror it around the x-axis, so it has a negative angle
    # Taking the modulo of a full circle forces the resulting angle to be between 0 and 2 pi
    angle = (acos * q) % (2 * np.pi)

    # If necessary, convert the angle in radians to degrees
    if kind.lower() in ('deg', 'degrees'):
        angle = angle * 180 / np.pi

    return angle


def convert_to_triangular_variables(df, column, kind='deg', remove_original_column=False):
    """
    Convert the columns in the given dataframe to triangular variables

    :param df: Pandas DataFrame
    :param column: column containing angular data to convert
    :param kind: specifying if the angular data is in degrees or in radians
    :param remove_original_column: whether to remove the original column from the dataframe

    :return: The data frame containing two new columns containing the triangular variables

    """
    df = df.copy()
    df[f'{column}_s'] = s_func(df[column], kind=kind)
    df[f'{column}_c'] = c_func(df[column], kind=kind)

    if remove_original_column:
        df = df.drop(columns=column)
    return df


def convert_to_trigonometric_variables(df, column, kind='deg', remove_original_column=False):
    """
    Convert the columns in the given dataframe to trigonometric variables

    :param df: Pandas DataFrame
    :param column: column containing angular data to convert
    :param kind: specifying if the angular data is in degrees or in radians
    :param remove_original_column: whether to remove the original column from the dataframe

    :return: The data frame containing two new columns containing the trigonometric variables

    """
    df = df.copy()

    if kind.lower() in ('deg', 'degree'):
        factor = np.pi / 180
    else:
        factor = 1
    df[f'{column}_sin'] = np.sin(df[column] * factor)
    df[f'{column}_cos'] = np.cos(df[column] * factor)

    if remove_original_column:
        df = df.drop(columns=column)

    return df


def convert_to_u_v(direction, speed, conversion='trigonometric', kind='deg'):
    """
    Convert a direction and speed variable to u and v variables.

    The u and v variables take the circular nature of the direction into account, as suggested by 3E. You can choose to
    use trigonometric or triangular functions to do the conversion

    :param direction: numpy array or pandas series containing the direction (angular) variable
    :param speed: numpy array or pandas series of the same length containing the speed (non-angular) variable
    :param conversion: convert using 'trigonometric' or 'triangular' functions
    :param kind: specifying if the angular data is in degrees or in radians

    :return: u and v arrays (or Series)

    """
    if kind.lower() in ('deg', 'degree'):
        factor = np.pi / 180
    else:
        factor = 1

    assert conversion in ('trigonometric', 'triangular')
    if conversion == 'trigonometric':
        u = np.sin(direction * factor) * speed
        v = np.cos(direction * factor) * speed
    else:
        u = s_func(direction, kind=kind) * speed
        v = c_func(direction, kind=kind) * speed

    return u, v


def u_v_to_direction_speed(u, v, conversion='trigonometric', kind='deg'):
    """
    Function that provides 2 arrays (or series) containing the direction and speed

    :param u: numpy array or pandas series containing the u variable
    :param v: numpy array or pandas series of the same length containing the v variable
    :param conversion: convert using 'trigonometric' or 'triangular' functions
    :param kind: specifying if the angular data is in degrees or in radians

    :return: 2 arrays (or series) containing the direction and speed

    """
    assert conversion in ('trigonometric', 'triangular')

    if conversion == 'trigonometric':
        speed = np.sqrt(u**2 + v**2)
        sin = u / speed
        cos = v / speed
        direction = revert_sin_cos_to_degrees(sin, cos, kind=kind)

    else:
        speed = np.abs(u) + np.abs(v)
        s = u / speed
        c = v / speed
        direction = revert_s_c_to_degrees(s, c, kind=kind)

    return direction, speed
