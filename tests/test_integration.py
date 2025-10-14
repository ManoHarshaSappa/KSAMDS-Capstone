"""
Integration tests for database operations
"""

import pytest
import psycopg2
import os


@pytest.fixture
def db_connection():
    """Create database connection for testing"""
    conn = psycopg2.connect(
        host=os.getenv('DB_HOST', 'localhost'),
        port=os.getenv('DB_PORT', '5432'),
        database=os.getenv('DB_NAME', 'ksamds_test'),
        user=os.getenv('DB_USER', 'postgres'),
        password=os.getenv('DB_PASSWORD', 'test_password')
    )
    yield conn
    conn.close()


class TestDatabaseIntegration:
    """Integration tests with actual database"""

    def test_database_connection(self, db_connection):
        """Test that we can connect to the database"""
        cursor = db_connection.cursor()
        cursor.execute('SELECT 1')
        result = cursor.fetchone()
        assert result[0] == 1

    def test_schema_exists(self, db_connection):
        """Test that ksamds schema exists"""
        cursor = db_connection.cursor()
        cursor.execute("""
            SELECT schema_name 
            FROM information_schema.schemata 
            WHERE schema_name = 'ksamds'
        """)
        result = cursor.fetchone()
        assert result is not None

    def test_core_tables_exist(self, db_connection):
        """Test that core tables exist in database"""
        cursor = db_connection.cursor()
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'ksamds'
        """)
        tables = [row[0] for row in cursor.fetchall()]

        required_tables = ['knowledge', 'skill', 'ability', 'occupation']
        for table in required_tables:
            assert table in tables
