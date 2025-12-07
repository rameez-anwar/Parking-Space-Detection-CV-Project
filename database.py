#!/usr/bin/env python3
"""
Database module for Parking Space Detection System
Handles SQLite operations for storing parking statistics and detection history
"""

import sqlite3
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import threading

class ParkingDatabase:
    def __init__(self, db_path: str = 'parking_data.db'):
        self.db_path = db_path
        self.lock = threading.Lock()
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Parking lots table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS parking_lots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    camera_url TEXT NOT NULL,
                    regions_file TEXT,
                    is_active BOOLEAN DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Detection sessions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS detection_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    parking_lot_id INTEGER,
                    session_start TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    session_end TIMESTAMP,
                    total_detections INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'active',
                    FOREIGN KEY (parking_lot_id) REFERENCES parking_lots (id)
                )
            ''')
            
            # Parking statistics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS parking_statistics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    parking_lot_id INTEGER,
                    session_id INTEGER,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    total_spaces INTEGER,
                    empty_spaces INTEGER,
                    filled_spaces INTEGER,
                    region_data TEXT,
                    image_data TEXT,
                    processed_image_data TEXT,
                    FOREIGN KEY (parking_lot_id) REFERENCES parking_lots (id),
                    FOREIGN KEY (session_id) REFERENCES detection_sessions (id)
                )
            ''')
            
            # System configuration table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS system_config (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Insert default configuration
            cursor.execute('''
                INSERT OR IGNORE INTO system_config (key, value) 
                VALUES ('detection_interval', '15')
            ''')
            
            # Migrate existing data if needed
            self._migrate_existing_data(cursor)
            
            # Insert default parking lots if none exist
            cursor.execute('SELECT COUNT(*) FROM parking_lots')
            if cursor.fetchone()[0] == 0:
                cursor.execute('''
                    INSERT INTO parking_lots (name, camera_url, regions_file)
                    VALUES ('Al Latif Plaza Parking', 'http://170.249.152.2:8080/video.mjpg', NULL)
                ''')
                cursor.execute('''
                    INSERT INTO parking_lots (name, camera_url, regions_file)
                    VALUES ('Cantt Plaza Parking', 'http://170.249.152.2:8080/video.mjpg', NULL)
                ''')
            
            conn.commit()
    
    def _migrate_existing_data(self, cursor):
        """Migrate existing data to new schema"""
        try:
            # Check if parking_lot_id column exists in detection_sessions
            cursor.execute("PRAGMA table_info(detection_sessions)")
            columns = [column[1] for column in cursor.fetchall()]
            if 'parking_lot_id' not in columns:
                cursor.execute('ALTER TABLE detection_sessions ADD COLUMN parking_lot_id INTEGER')
                print("Added parking_lot_id column to detection_sessions table")
            
            # Check if parking_lot_id column exists in parking_statistics
            cursor.execute("PRAGMA table_info(parking_statistics)")
            columns = [column[1] for column in cursor.fetchall()]
            if 'parking_lot_id' not in columns:
                cursor.execute('ALTER TABLE parking_statistics ADD COLUMN parking_lot_id INTEGER')
                print("Added parking_lot_id column to parking_statistics table")
                
                # If there's existing data, assign it to the first parking lot
                cursor.execute('SELECT COUNT(*) FROM parking_statistics WHERE parking_lot_id IS NULL')
                if cursor.fetchone()[0] > 0:
                    cursor.execute('SELECT id FROM parking_lots LIMIT 1')
                    first_lot = cursor.fetchone()
                    if first_lot:
                        cursor.execute('UPDATE parking_statistics SET parking_lot_id = ? WHERE parking_lot_id IS NULL', (first_lot[0],))
                        print(f"Migrated existing statistics to parking lot {first_lot[0]}")
            
        except Exception as e:
            print(f"Migration error: {e}")
            # Continue anyway - the app should still work
    
    def get_parking_lots(self) -> List[Dict]:
        """Get all parking lots"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, name, camera_url, regions_file, is_active, created_at, updated_at
                FROM parking_lots
                ORDER BY name
            ''')
            results = []
            for row in cursor.fetchall():
                results.append({
                    'id': row[0],
                    'name': row[1],
                    'camera_url': row[2],
                    'regions_file': row[3],
                    'is_active': bool(row[4]),
                    'created_at': row[5],
                    'updated_at': row[6]
                })
            return results
    
    def add_parking_lot(self, name: str, camera_url: str, regions_file: str = None) -> int:
        """Add a new parking lot"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO parking_lots (name, camera_url, regions_file)
                VALUES (?, ?, ?)
            ''', (name, camera_url, regions_file))
            lot_id = cursor.lastrowid
            conn.commit()
            return lot_id
    
    def update_parking_lot(self, lot_id: int, name: str = None, camera_url: str = None, 
                          regions_file: str = None, is_active: bool = None):
        """Update a parking lot"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            updates = []
            params = []
            
            if name is not None:
                updates.append('name = ?')
                params.append(name)
            if camera_url is not None:
                updates.append('camera_url = ?')
                params.append(camera_url)
            if regions_file is not None:
                updates.append('regions_file = ?')
                params.append(regions_file)
            if is_active is not None:
                updates.append('is_active = ?')
                params.append(is_active)
            
            if updates:
                updates.append('updated_at = ?')
                params.append(datetime.now())
                params.append(lot_id)
                
                cursor.execute(f'''
                    UPDATE parking_lots 
                    SET {', '.join(updates)}
                    WHERE id = ?
                ''', params)
                conn.commit()
    
    def delete_parking_lot(self, lot_id: int):
        """Delete a parking lot"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM parking_lots WHERE id = ?', (lot_id,))
            conn.commit()
    
    def start_session(self, parking_lot_id: int = None) -> int:
        """Start a new detection session"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO detection_sessions (parking_lot_id, session_start, status)
                VALUES (?, ?, 'active')
            ''', (parking_lot_id, datetime.now()))
            session_id = cursor.lastrowid
            conn.commit()
            return session_id
    
    def end_session(self, session_id: int):
        """End a detection session"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE detection_sessions 
                SET session_end = ?, status = 'completed'
                WHERE id = ?
            ''', (datetime.now(), session_id))
            conn.commit()
    
    def save_detection_data(self, parking_lot_id: int, session_id: int, total_spaces: int, 
                          empty_spaces: int, filled_spaces: int, 
                          region_data: Dict, image_data: str = None, 
                          processed_image_data: str = None):
        """Save detection statistics to database"""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check if parking_lot_id column exists
                cursor.execute("PRAGMA table_info(parking_statistics)")
                columns = [column[1] for column in cursor.fetchall()]
                
                # Convert region data to JSON string
                region_json = json.dumps(region_data)
                
                if 'parking_lot_id' in columns:
                    # New schema with parking_lot_id
                    cursor.execute('''
                        INSERT INTO parking_statistics 
                        (parking_lot_id, session_id, total_spaces, empty_spaces, filled_spaces, 
                         region_data, image_data, processed_image_data)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (parking_lot_id, session_id, total_spaces, empty_spaces, filled_spaces,
                          region_json, image_data, processed_image_data))
                else:
                    # Old schema without parking_lot_id
                    cursor.execute('''
                        INSERT INTO parking_statistics 
                        (session_id, total_spaces, empty_spaces, filled_spaces, 
                         region_data, image_data, processed_image_data)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (session_id, total_spaces, empty_spaces, filled_spaces,
                          region_json, image_data, processed_image_data))
                
                # Update session detection count
                cursor.execute('''
                    UPDATE detection_sessions 
                    SET total_detections = total_detections + 1
                    WHERE id = ?
                ''', (session_id,))
                
                conn.commit()
    
    def get_latest_statistics(self, parking_lot_id: int = None) -> Optional[Dict]:
        """Get the most recent parking statistics for a specific lot or all lots"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            if parking_lot_id:
                cursor.execute('''
                    SELECT p.name, ps.total_spaces, ps.empty_spaces, ps.filled_spaces, 
                           ps.region_data, ps.timestamp, ps.image_data, ps.processed_image_data
                    FROM parking_statistics ps
                    JOIN parking_lots p ON ps.parking_lot_id = p.id
                    WHERE ps.parking_lot_id = ?
                    ORDER BY ps.timestamp DESC 
                    LIMIT 1
                ''', (parking_lot_id,))
            else:
                cursor.execute('''
                    SELECT p.name, ps.total_spaces, ps.empty_spaces, ps.filled_spaces, 
                           ps.region_data, ps.timestamp, ps.image_data, ps.processed_image_data
                    FROM parking_statistics ps
                    JOIN parking_lots p ON ps.parking_lot_id = p.id
                    ORDER BY ps.timestamp DESC 
                    LIMIT 1
                ''')
            
            row = cursor.fetchone()
            if row:
                return {
                    'parking_lot_name': row[0],
                    'total_spaces': row[1],
                    'empty_spaces': row[2],
                    'filled_spaces': row[3],
                    'region_breakdown': json.loads(row[4]) if row[4] else {},
                    'last_update': row[5],
                    'image_data': row[6],
                    'processed_image_data': row[7]
                }
            return None
    
    def get_all_latest_statistics(self) -> List[Dict]:
        """Get the most recent statistics for all parking lots"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Check if parking_lot_id column exists
            cursor.execute("PRAGMA table_info(parking_statistics)")
            columns = [column[1] for column in cursor.fetchall()]
            
            if 'parking_lot_id' not in columns:
                # Fallback for old schema - return empty results
                print("Warning: parking_lot_id column not found, returning empty statistics")
                return []
            
            try:
                cursor.execute('''
                    SELECT p.id, p.name, p.camera_url, p.is_active,
                           ps.total_spaces, ps.empty_spaces, ps.filled_spaces, 
                           ps.region_data, ps.timestamp
                    FROM parking_lots p
                    LEFT JOIN (
                        SELECT parking_lot_id, total_spaces, empty_spaces, filled_spaces, 
                               region_data, timestamp,
                               ROW_NUMBER() OVER (PARTITION BY parking_lot_id ORDER BY timestamp DESC) as rn
                        FROM parking_statistics
                    ) ps ON p.id = ps.parking_lot_id AND ps.rn = 1
                    WHERE p.is_active = 1
                    ORDER BY p.name
                ''')
                
                results = []
                for row in cursor.fetchall():
                    results.append({
                        'id': row[0],
                        'name': row[1],
                        'camera_url': row[2],
                        'is_active': bool(row[3]),
                        'total_spaces': row[4] or 0,
                        'empty_spaces': row[5] or 0,
                        'filled_spaces': row[6] or 0,
                        'region_breakdown': json.loads(row[7]) if row[7] else {},
                        'last_update': row[8] or 'No data'
                    })
                return results
            except Exception as e:
                print(f"Error getting statistics: {e}")
                return []
    
    def get_statistics_history(self, limit: int = 100) -> List[Dict]:
        """Get historical statistics"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT total_spaces, empty_spaces, filled_spaces, 
                       region_data, timestamp
                FROM parking_statistics 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (limit,))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    'total_spaces': row[0],
                    'empty_spaces': row[1],
                    'filled_spaces': row[2],
                    'region_breakdown': json.loads(row[3]) if row[3] else {},
                    'timestamp': row[4]
                })
            return results
    
    def get_active_session(self) -> Optional[int]:
        """Get the current active session ID"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id FROM detection_sessions 
                WHERE status = 'active' 
                ORDER BY session_start DESC 
                LIMIT 1
            ''')
            row = cursor.fetchone()
            return row[0] if row else None
    
    def get_system_config(self, key: str) -> str:
        """Get system configuration value"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT value FROM system_config WHERE key = ?
            ''', (key,))
            row = cursor.fetchone()
            return row[0] if row else None
    
    def update_system_config(self, key: str, value: str):
        """Update system configuration"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO system_config (key, value, updated_at)
                VALUES (?, ?, ?)
            ''', (key, value, datetime.now()))
            conn.commit()
    
    def cleanup_old_data(self, days_to_keep: int = 30):
        """Clean up old detection data"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                DELETE FROM parking_statistics 
                WHERE timestamp < datetime('now', '-{} days')
            '''.format(days_to_keep))
            
            cursor.execute('''
                DELETE FROM detection_sessions 
                WHERE session_start < datetime('now', '-{} days')
            '''.format(days_to_keep))
            
            conn.commit()
    
    def get_session_summary(self, session_id: int) -> Optional[Dict]:
        """Get summary statistics for a session"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT 
                    AVG(empty_spaces) as avg_empty,
                    MAX(empty_spaces) as max_empty,
                    MIN(empty_spaces) as min_empty,
                    COUNT(*) as total_readings,
                    MIN(timestamp) as session_start,
                    MAX(timestamp) as session_end
                FROM parking_statistics 
                WHERE session_id = ?
            ''', (session_id,))
            
            row = cursor.fetchone()
            if row:
                return {
                    'avg_empty_spaces': round(row[0], 2) if row[0] else 0,
                    'max_empty_spaces': row[1] or 0,
                    'min_empty_spaces': row[2] or 0,
                    'total_readings': row[3] or 0,
                    'session_start': row[4],
                    'session_end': row[5]
                }
            return None
