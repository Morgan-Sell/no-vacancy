import os
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool

from alembic import context as alembic_context  # type: ignore[attr-defined]
from app.db.postgres import AsyncPostgresDB  # only for URL generation
from app.schemas import bronze, gold, silver

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = alembic_context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Approach is required for Medallion architecture w/ separate databases
# Target DB defaults to bronze
target_db = os.environ.get("ALEMBIC_TARGET_DB", "bronze")

df_map = {
    "bronze": (
        bronze.BronzeBase.metadata,
        AsyncPostgresDB(
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            host=os.getenv("BRONZE_DB_HOST"),
            port=os.getenv("DB_PORT"),
            db_name=os.getenv("BRONZE_DB"),
        ).build_url(async_mode=False),
    ),
    "silver": (
        silver.SilverBase.metadata,
        AsyncPostgresDB(
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            host=os.getenv("SILVER_DB_HOST"),
            port=os.getenv("DB_PORT"),
            db_name=os.getenv("SILVER_DB"),
        ).build_url(async_mode=False),
    ),
    "gold": (
        gold.GoldBase.metadata,
        AsyncPostgresDB(
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            host=os.getenv("GOLD_DB_HOST"),
            port=os.getenv("DB_PORT"),
            db_name=os.getenv("GOLD_DB"),
        ).build_url(async_mode=False),
    ),
}

try:
    target_metadata, sqlalchemy_url = df_map[target_db]
except KeyError as err:
    raise ValueError(
        f"Invalid target database: {target_db}. Must be one of {list(df_map.keys())}."
    ) from err

# Set dynamic database URL
config.set_main_option("sqlalchemy.url", sqlalchemy_url)


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    alembic_context.configure(
        url=sqlalchemy_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with alembic_context.begin_transaction():
        alembic_context.run_migrations()


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
        alembic_context.configure(
            connection=connection, target_metadata=target_metadata
        )

        with alembic_context.begin_transaction():
            alembic_context.run_migrations()


if alembic_context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
