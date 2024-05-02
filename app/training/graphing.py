from .constants import FrameData


def calculate_line_equation(point1: FrameData, point2: FrameData) -> str:
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


def calculate_graphs(data: list):
    for i in range(len(data)):  # every frame
        for j in range(len(data[i])):  # every landmark
            if i == 0:
                data[i][j].direct_graph = "0"
                continue
            data[i][j].direct_graph = calculate_line_equation(data[i - 1][j], data[i][j])