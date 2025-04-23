import os
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool, MetaData

from alembic import context

from app.db.postgres import (
    BRONZE_DB_URL,
    SILVER_DB_URL,
    GOLD_DB_URL,
)

from app.schemas.bronze import Base as bronze_base
from app.schemas.silver import Base as silver_base
from app.schemas.gold import Base as gold_base


# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# add your model's MetaData object here
# for 'autogenerate' support
# from myapp import mymodel
# target_metadata = mymodel.Base.metadata

# Approach is required for Medallion architecture w/ separate databases
target_db = os.environ.get("ALEMBIC_TARGET_DB", "bronze")

df_config_map = {
    "bronze": (BRONZE_DB_URL, bronze_base.metadata),
    "silver": (SILVER_DB_URL, silver_base.metadata),
    "gold": (GOLD_DB_URL, gold_base.metadata),
}

try:
    sqlalchemy_url, target_metadata = df_config_map[target_db]
except KeyError:
    raise ValueError(
        f"Invalid target database: {target_db}. Must be one of {list(df_config_map.keys())}."
    )

config.set_main_option("sqlalchemy.url", sqlalchemy_url)

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
