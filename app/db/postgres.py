from config import DB_CONNECT_TIMEOUT
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


class PostgresDB:
    def __init__(self, user, password, host, port, db_name):
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        self.db_name = db_name
        print(
            f"[DEBUG] Creating PostgresDB with: {user=}, {password=}, {host=}, {port=}, {db_name=}"
        )
        self.port = int(port)

        self.url = self._build_url()
        self.engine = self._create_engine()
        self.SessionLocal = self._create_session()

    def _build_url(self):
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.db_name}"

    def _create_engine(self):
        return create_engine(
            self.url, connect_args={"connect_timeout": DB_CONNECT_TIMEOUT}
        )

    def _create_session(self):
        return sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
