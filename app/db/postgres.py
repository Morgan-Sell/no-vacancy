from config import DB_CONNECT_TIMEOUT
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker


class AsyncPostgresDB:
    def __init__(self, user, password, host, port, db_name):
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        self.db_name = db_name
        self.port = int(port)

        self.url = self._build_url()
        self.engine = self._create_engine()
        self.SessionLocal = self._create_session()

    def _build_url(self):
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.db_name}"

    def _create_engine(self):
        return create_async_engine(
            self.url, echo=False, connect_args={"timeout": DB_CONNECT_TIMEOUT}
        )

    def _create_session(self):
        return sessionmaker(
            bind=self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
