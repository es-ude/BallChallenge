class Board:
    def __init__(self, picture):
        self.corners: tuple[tuple[int, int],tuple[int, int],tuple[int, int],tuple[int, int]] = self._detect_corners(picture)

    def _detect_corners(self, picture):
        ...

    def is_point_in_board(self, point: tuple[int, int]) -> bool:
        x_min = min(self.corners[0][0], self.corners[1][0], self.corners[2][0], self.corners[3][0])
        x_max = max(self.corners[0][0], self.corners[1][0], self.corners[2][0], self.corners[3][0])
        y_min = min(self.corners[0][1], self.corners[1][1], self.corners[2][1], self.corners[3][1])
        y_max = max(self.corners[0][1], self.corners[1][1], self.corners[2][1], self.corners[3][1])
        if point[0] >= x_min and point[0] <= x_max:
            if point[1] >= y_min and point[1] <= y_max:
                return True
        return False
