from dataclasses import dataclass
from math import ceil


TEMPLATE_VERSION = "liensina-omr-v1"
CANONICAL_WIDTH = 1100
CANONICAL_HEIGHT = 1550
MARKER_SIZE = 58
MARKER_MARGIN = 54
GRID_TOP = 360
GRID_BOTTOM_MARGIN = 120
GRID_LEFT = 88
OPTIONS = ("A", "B", "C", "D", "E")


@dataclass(frozen=True)
class BubblePosition:
    question_number: int
    option: str
    x: int
    y: int
    radius: int


@dataclass(frozen=True)
class GridLayout:
    total_questions: int
    columns: int
    rows_per_column: int
    column_width: float
    column_step: float
    grid_left: float
    question_label_width: float
    first_bubble_offset: float
    row_height: float
    bubble_spacing: float
    bubble_radius: int


def build_grid_layout(total_questions: int) -> GridLayout:
    columns = min(4, max(1, ceil(total_questions / 25)))
    rows_per_column = ceil(total_questions / columns)
    available_height = CANONICAL_HEIGHT - GRID_TOP - GRID_BOTTOM_MARGIN
    row_height = min(48.0, max(30.0, available_height / max(rows_per_column, 1)))
    question_label_width = 48.0 if columns == 1 else 42.0 if columns == 2 else 36.0 if columns == 3 else 32.0
    question_label_gap = 34.0 if columns == 1 else 24.0 if columns == 2 else 18.0 if columns == 3 else 14.0
    column_gap = 0.0 if columns == 1 else 72.0 if columns == 2 else 50.0 if columns == 3 else 28.0
    base_bubble_spacing = 68.0 if columns == 1 else 52.0 if columns == 2 else 42.0 if columns == 3 else 36.0
    max_bubble_radius = 22 if columns == 1 else 19 if columns == 2 else 16 if columns == 3 else 14
    bubble_radius = min(max_bubble_radius, max(13, int(row_height * 0.42)))
    bubble_spacing = max(bubble_radius * 2 + 8.0, base_bubble_spacing)
    option_block_width = (len(OPTIONS) - 1) * bubble_spacing + bubble_radius * 2
    column_width = question_label_width + question_label_gap + option_block_width
    grid_width = columns * column_width + (columns - 1) * column_gap
    grid_left = (CANONICAL_WIDTH - grid_width) / 2
    first_bubble_offset = question_label_width + question_label_gap + bubble_radius
    column_step = column_width + column_gap
    return GridLayout(
        total_questions=total_questions,
        columns=columns,
        rows_per_column=rows_per_column,
        column_width=column_width,
        column_step=column_step,
        grid_left=grid_left,
        question_label_width=question_label_width,
        first_bubble_offset=first_bubble_offset,
        row_height=row_height,
        bubble_spacing=bubble_spacing,
        bubble_radius=bubble_radius,
    )


def bubble_positions(total_questions: int) -> list[BubblePosition]:
    layout = build_grid_layout(total_questions)
    positions: list[BubblePosition] = []
    for question_index in range(total_questions):
        question_number = question_index + 1
        column = question_index // layout.rows_per_column
        row = question_index % layout.rows_per_column
        column_x = layout.grid_left + column * layout.column_step
        row_y = GRID_TOP + row * layout.row_height
        first_bubble_x = column_x + layout.first_bubble_offset
        for option_index, option in enumerate(OPTIONS):
            positions.append(
                BubblePosition(
                    question_number=question_number,
                    option=option,
                    x=int(round(first_bubble_x + option_index * layout.bubble_spacing)),
                    y=int(round(row_y)),
                    radius=layout.bubble_radius,
                )
            )
    return positions


def marker_centers() -> list[tuple[float, float]]:
    offset = MARKER_MARGIN + MARKER_SIZE / 2
    return [
        (offset, offset),
        (CANONICAL_WIDTH - offset, offset),
        (CANONICAL_WIDTH - offset, CANONICAL_HEIGHT - offset),
        (offset, CANONICAL_HEIGHT - offset),
    ]
