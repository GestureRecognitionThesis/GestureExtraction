from .constants import FrameData


def calculate_line_equation(point1: FrameData, point2: FrameData, id: int = 0) -> str:
    # x = 0, y = 1, z = 2
    c_point1: FrameData = point1
    c_point2: FrameData = point2

    if point1.relative[0] > point2.relative[0]:
        c_point1 = point2
        c_point2 = point1

    # Calculate the slope
    try:
        slope = (c_point2.relative[1] - c_point1.relative[1]) / (c_point2.relative[0] - c_point1.relative[0])
    except ZeroDivisionError:
        return "0"

    # Calculate the y-intercept
    y_intercept = c_point1.relative[1] - slope * point1.relative[0]

    result: str = f"{slope}x + {y_intercept}"

    return result
