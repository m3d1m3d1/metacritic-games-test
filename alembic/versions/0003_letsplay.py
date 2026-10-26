"""store letsplay summaries

Revision ID: 0003
Revises: 0002
Create Date: 2026-10-22
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "0003"
down_revision: Union[str, Sequence[str], None] = "0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("games", sa.Column("letsplay_url", sa.Text(), nullable=True))
    op.add_column("games", sa.Column("letsplay_title", sa.Text(), nullable=True))
    op.add_column("games", sa.Column("letsplay_views", sa.Integer(), nullable=True))
    op.add_column("games", sa.Column("letsplay_summary", sa.Text(), nullable=True))
    op.add_column("games", sa.Column("letsplay_source_hash", sa.String(length=64), nullable=True))


def downgrade() -> None:
    op.drop_column("games", "letsplay_source_hash")
    op.drop_column("games", "letsplay_summary")
    op.drop_column("games", "letsplay_views")
    op.drop_column("games", "letsplay_title")
    op.drop_column("games", "letsplay_url")
