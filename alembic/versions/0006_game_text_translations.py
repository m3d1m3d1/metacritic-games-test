"""cache Russian translations for game descriptions and summaries

Revision ID: 0006
Revises: 0005
Create Date: 2026-10-26
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "0006"
down_revision: Union[str, Sequence[str], None] = "0005"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("games", sa.Column("description_ru", sa.Text(), nullable=True))
    op.add_column("games", sa.Column("letsplay_summary_ru", sa.Text(), nullable=True))
    op.add_column("review_summaries", sa.Column("summary_ru", sa.Text(), nullable=True))
    op.add_column("review_summaries", sa.Column("likes_ru", sa.ARRAY(sa.Text()), nullable=True))
    op.add_column("review_summaries", sa.Column("dislikes_ru", sa.ARRAY(sa.Text()), nullable=True))


def downgrade() -> None:
    op.drop_column("review_summaries", "dislikes_ru")
    op.drop_column("review_summaries", "likes_ru")
    op.drop_column("review_summaries", "summary_ru")
    op.drop_column("games", "letsplay_summary_ru")
    op.drop_column("games", "description_ru")
