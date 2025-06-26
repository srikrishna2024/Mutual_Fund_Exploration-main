import subprocess
import os
import sys
import re
from datetime import datetime

class PostgreSQLMigrator:
    def __init__(self):
        # Configurable parameters with defaults
        self.db_params = {
            'dbname': 'postgres',
            'user': 'postgres',
            'password': 'admin123',
            'host': 'localhost',
            'port': '5432'
        }
        
        # Version information
        self.source_version = None
        self.target_version = None
        
        # Try to auto-detect PostgreSQL binary paths
        self.pg_bin_path = self.detect_postgresql_bin_path()
        self.target_version = self.get_postgresql_version()
        
        # Migration options
        self.backup_types = {
            '1': 'Single database (custom format)',
            '2': 'All databases (pg_dumpall)',
            '3': 'Roles only',
            '4': 'Single database (plain SQL)'
        }

    def detect_postgresql_bin_path(self):
        """Try to auto-detect PostgreSQL binary path."""
        possible_paths = [
            r'C:\Program Files\PostgreSQL\17\bin',
            r'C:\Program Files\PostgreSQL\16\bin',
            r'C:\Program Files\PostgreSQL\15\bin',
            r'C:\Program Files\PostgreSQL\14\bin',
            r'C:\Program Files\PostgreSQL\13\bin',
        ]
        
        for path in possible_paths:
            if os.path.isdir(path):
                return path
                
        if self.is_command_available('psql'):
            return None
        
        return None

    def get_postgresql_version(self):
        """Get the PostgreSQL version of the target server."""
        try:
            psql = self.get_full_path('psql')
            cmd = [
                psql,
                '-h', self.db_params['host'],
                '-p', self.db_params['port'],
                '-U', self.db_params['user'],
                '-c', "SELECT version();"
            ]
            
            env = os.environ.copy()
            env['PGPASSWORD'] = self.db_params['password']
            
            result = subprocess.run(cmd, env=env, 
                                 stdout=subprocess.PIPE, 
                                 stderr=subprocess.PIPE,
                                 text=True,
                                 check=True)
            
            version_line = result.stdout.split('\n')[2]
            version_match = re.search(r'PostgreSQL (\d+\.\d+)', version_line)
            if version_match:
                return version_match.group(1)
            return None
        except Exception as e:
            print(f"Warning: Could not determine PostgreSQL version: {e}")
            return None

    def display_version_info(self, operation):
        """Display version information for the current operation."""
        print("\n" + "="*60)
        print(f" {operation.upper()} VERSION INFORMATION")
        print("="*60)
        if self.target_version:
            print(f" Target Server Version: {self.target_version}")
        else:
            print(" Target Server Version: Unknown")
        
        if self.source_version:
            print(f" Backup Source Version: {self.source_version}")
        
        if self.source_version and self.target_version:
            source_major = self.source_version.split('.')[0]
            target_major = self.target_version.split('.')[0]
            
            if source_major != target_major:
                print("\n⚠️  MAJOR VERSION DIFFERENCE DETECTED!")
                print(f" Backup was created for PostgreSQL {self.source_version}")
                print(f" Target server is running PostgreSQL {self.target_version}")
                print("\nRecommendation: For major version upgrades, use:")
                print("- Plain SQL format (option 4 for backup)")
                print("- Test restore in staging environment first")
        print("="*60 + "\n")

    def get_valid_backup_folder(self):
        """Get and validate a backup folder path with proper path handling."""
        while True:
            folder = input("Enter folder path to save backup: ").strip()
            try:
                # Normalize path (handles both / and \ correctly)
                folder = os.path.normpath(folder)
                
                # Convert to absolute path
                folder = os.path.abspath(folder)
                
                # Create directory if it doesn't exist
                os.makedirs(folder, exist_ok=True)
                
                # Verify it's a directory and writable
                if not os.path.isdir(folder):
                    raise ValueError("Path is not a directory")
                if not os.access(folder, os.W_OK):
                    raise ValueError("Directory is not writable")
                    
                return folder
            except Exception as e:
                print(f"Invalid backup folder: {e}. Please try again.")

    def is_command_available(self, command):
        """Check if a command is available in the system PATH."""
        try:
            subprocess.run([command, '--version'], 
                          stdout=subprocess.PIPE, 
                          stderr=subprocess.PIPE,
                          check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def get_full_path(self, command):
        """Get full path to PostgreSQL command."""
        if self.pg_bin_path:
            return os.path.join(self.pg_bin_path, command)
        return command

    def configure_parameters(self):
        """Allow user to configure connection parameters."""
        print("\nCurrent PostgreSQL connection parameters:")
        for key, value in self.db_params.items():
            print(f"{key}: {value}")
            
        change = input("\nDo you want to change these parameters? (y/n): ").lower()
        if change != 'y':
            return
            
        print("\nEnter new parameters (press Enter to keep current value):")
        for key in self.db_params:
            new_value = input(f"{key} [{self.db_params[key]}]: ").strip()
            if new_value:
                self.db_params[key] = new_value
        
        # Update target version after parameter changes
        self.target_version = self.get_postgresql_version()

    def create_backup(self):
        """Create a backup of PostgreSQL database(s)."""
        self.display_version_info("backup")
        print("\nBackup Options:")
        for key, desc in self.backup_types.items():
            print(f"{key}. {desc}")
            
        choice = input("Enter your choice: ").strip()
        
        if choice not in self.backup_types:
            print("Invalid choice.")
            return
            
        backup_folder = self.get_valid_backup_folder()
        current_date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        
        env = os.environ.copy()
        env['PGPASSWORD'] = self.db_params['password']
        
        try:
            if choice == '1':  # Single database (custom format)
                backup_file = os.path.join(backup_folder, f'db_{self.db_params["dbname"]}_backup_{current_date}.dump')
                pg_dump = self.get_full_path('pg_dump')
                
                cmd = [
                    pg_dump,
                    '-h', self.db_params['host'],
                    '-p', self.db_params['port'],
                    '-U', self.db_params['user'],
                    '-F', 'c',
                    '-f', backup_file,
                    '-v',
                    self.db_params['dbname']
                ]
                
                print(f"\nCreating backup compatible with PostgreSQL {self.target_version}...")
                subprocess.run(cmd, env=env, check=True)
                print(f"✅ Database backup created: {backup_file}")
                
            elif choice == '2':  # All databases (includes roles)
                backup_file = os.path.join(backup_folder, f'full_cluster_backup_{current_date}.sql')
                pg_dumpall = self.get_full_path('pg_dumpall')
                
                cmd = [
                    pg_dumpall,
                    '-h', self.db_params['host'],
                    '-p', self.db_params['port'],
                    '-U', self.db_params['user'],
                    '-f', backup_file,
                    '-v'
                ]
                
                print(f"\nCreating full cluster backup compatible with PostgreSQL {self.target_version}...")
                subprocess.run(cmd, env=env, check=True)
                print(f"✅ Full cluster backup created: {backup_file}")
                
            elif choice == '3':  # Roles only
                roles_file = os.path.join(backup_folder, f'roles_backup_{current_date}.sql')
                self.backup_roles(roles_file)
                print(f"✅ Roles backup created: {roles_file}")
                
            elif choice == '4':  # Single database (plain SQL)
                backup_file = os.path.join(backup_folder, f'db_{self.db_params["dbname"]}_backup_{current_date}.sql')
                pg_dump = self.get_full_path('pg_dump')
                
                cmd = [
                    pg_dump,
                    '-h', self.db_params['host'],
                    '-p', self.db_params['port'],
                    '-U', self.db_params['user'],
                    '-F', 'p',
                    '-f', backup_file,
                    '-v',
                    self.db_params['dbname']
                ]
                
                print(f"\nCreating SQL-format backup compatible with PostgreSQL {self.target_version}...")
                subprocess.run(cmd, env=env, check=True)
                print(f"✅ Database backup created: {backup_file}")
                
        except subprocess.CalledProcessError as e:
            print(f"❌ Backup failed: {e}")
            return False
            
        return True

    def backup_roles(self, output_file):
        """Backup database roles only."""
        pg_dumpall = self.get_full_path('pg_dumpall')
        
        cmd = [
            pg_dumpall,
            '-h', self.db_params['host'],
            '-p', self.db_params['port'],
            '-U', self.db_params['user'],
            '--roles-only',
            '-f', output_file
        ]
        
        env = os.environ.copy()
        env['PGPASSWORD'] = self.db_params['password']
        
        print("\nBacking up roles...")
        subprocess.run(cmd, env=env, check=True)

    def detect_backup_version(self, backup_file):
        """Detect PostgreSQL version from backup file."""
        self.source_version = None
        try:
            if backup_file.endswith('.dump'):
                pg_restore = self.get_full_path('pg_restore')
                result = subprocess.run(
                    [pg_restore, '--list', backup_file],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                for line in result.stderr.splitlines():
                    if 'built for PostgreSQL' in line:
                        self.source_version = line.split()[-1]
                        break
            else:
                with open(backup_file, 'r', encoding='utf-8') as f:
                    for line in f.readlines()[:20]:
                        if 'PostgreSQL' in line and 'dump' in line:
                            version_part = line.split('PostgreSQL ')[1]
                            self.source_version = version_part.split()[0].strip(')')
                            break
        except Exception as e:
            print(f"Warning: Could not detect backup version - {e}")

    def restore_backup(self):
        """Restore a PostgreSQL backup."""
        self.display_version_info("restore")
        print("\nRestore Options:")
        print("1. Restore single database")
        print("2. Restore full cluster backup (pg_dumpall)")
        print("3. Restore roles only")
        
        choice = input("Enter your choice: ").strip()
        
        if choice not in ['1', '2', '3']:
            print("Invalid choice.")
            return
            
        backup_file = self.get_valid_path("Enter path to backup file: ")
        if not os.path.isfile(backup_file):
            print("Backup file not found.")
            return
            
        self.detect_backup_version(backup_file)
        self.display_version_info("restore")
        
        if self.source_version and self.target_version:
            source_major = self.source_version.split('.')[0]
            target_major = self.target_version.split('.')[0]
            
            if source_major != target_major:
                confirm = input("\n⚠️  Major version difference detected! Continue? (y/n): ").lower()
                if confirm != 'y':
                    return

        env = os.environ.copy()
        env['PGPASSWORD'] = self.db_params['password']
        
        try:
            if choice == '1':
                print("\nDatabase restore options:")
                print("1. Clean restore (drop existing database)")
                print("2. Restore to new database")
                
                restore_choice = input("Enter your choice: ").strip()
                
                if restore_choice == '1':
                    db_name = self.db_params['dbname']
                    pg_restore = self.get_full_path('pg_restore')
                    
                    cmd = [
                        pg_restore,
                        '-h', self.db_params['host'],
                        '-p', self.db_params['port'],
                        '-U', self.db_params['user'],
                        '--clean',
                        '--create',
                        '-d', 'postgres',
                        '-v',
                        backup_file
                    ]
                    
                    print(f"\nRestoring database '{db_name}'...")
                    subprocess.run(cmd, env=env, check=True)
                    
                elif restore_choice == '2':
                    db_name = input("\nEnter new database name: ").strip()
                    if not db_name:
                        print("Database name cannot be empty.")
                        return
                        
                    createdb = self.get_full_path('createdb')
                    cmd = [
                        createdb,
                        '-h', self.db_params['host'],
                        '-p', self.db_params['port'],
                        '-U', self.db_params['user'],
                        db_name
                    ]
                    
                    print(f"\nCreating new database '{db_name}'...")
                    subprocess.run(cmd, env=env, check=True)
                    
                    pg_restore = self.get_full_path('pg_restore')
                    cmd = [
                        pg_restore,
                        '-h', self.db_params['host'],
                        '-p', self.db_params['port'],
                        '-U', self.db_params['user'],
                        '-d', db_name,
                        '-v',
                        backup_file
                    ]
                    
                    print(f"Restoring to new database '{db_name}'...")
                    subprocess.run(cmd, env=env, check=True)
                    
                else:
                    print("Invalid choice.")
                    return
                    
                self.post_restore_actions(db_name if restore_choice == '2' else self.db_params['dbname'])
                print("\n✅ Database restore completed successfully.")
                
            elif choice == '2':
                psql = self.get_full_path('psql')
                cmd = [
                    psql,
                    '-h', self.db_params['host'],
                    '-p', self.db_params['port'],
                    '-U', self.db_params['user'],
                    '-f', backup_file,
                    '-v'
                ]
                
                print("\nRestoring full cluster backup...")
                subprocess.run(cmd, env=env, check=True)
                self.post_restore_full_cluster()
                print("\n✅ Full cluster restore completed successfully.")
                
            elif choice == '3':
                self.restore_roles(backup_file)
                print("\n✅ Roles restore completed successfully.")
                
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Restore failed: {e}")
            return False
            
        return True

    def get_valid_path(self, prompt):
        """Safely get and validate a filesystem path."""
        while True:
            path = input(prompt).strip()
            try:
                path = os.path.normpath(path)
                path = os.path.abspath(path)
                
                if not os.path.exists(path):
                    raise ValueError("Path does not exist")
                    
                return path
            except Exception as e:
                print(f"Invalid path: {e}. Please try again.")

    def restore_roles(self, roles_file):
        """Restore database roles from backup."""
        psql = self.get_full_path('psql')
        
        cmd = [
            psql,
            '-h', self.db_params['host'],
            '-p', self.db_params['port'],
            '-U', self.db_params['user'],
            '-f', roles_file,
            '-v'
        ]
        
        env = os.environ.copy()
        env['PGPASSWORD'] = self.db_params['password']
        
        print("\nRestoring roles...")
        subprocess.run(cmd, env=env, check=True)

    def post_restore_actions(self, db_name):
        """Perform post-restore maintenance tasks."""
        print("\nRunning post-restore maintenance...")
        
        env = os.environ.copy()
        env['PGPASSWORD'] = self.db_params['password']
        psql = self.get_full_path('psql')
        
        try:
            print("- Updating database statistics (ANALYZE)...")
            cmd = [
                psql,
                '-h', self.db_params['host'],
                '-p', self.db_params['port'],
                '-U', self.db_params['user'],
                '-d', db_name,
                '-c', "ANALYZE;"
            ]
            subprocess.run(cmd, env=env, check=True)
            
            print("- Rebuilding indexes (REINDEX)...")
            cmd = [
                psql,
                '-h', self.db_params['host'],
                '-p', self.db_params['port'],
                '-U', self.db_params['user'],
                '-d', db_name,
                '-c', f"REINDEX DATABASE {db_name};"
            ]
            subprocess.run(cmd, env=env, check=True)
            
            print("- Testing database connectivity...")
            cmd = [
                psql,
                '-h', self.db_params['host'],
                '-p', self.db_params['port'],
                '-U', self.db_params['user'],
                '-d', db_name,
                '-c', "SELECT 1 AS connection_test;"
            ]
            result = subprocess.run(cmd, env=env, 
                                  stdout=subprocess.PIPE, 
                                  stderr=subprocess.PIPE,
                                  text=True,
                                  check=True)
            if "1 row" in result.stdout:
                print("  ✅ Connectivity test passed")
            else:
                print("  ❌ Connectivity test failed")
                
            print("\nChecking extensions...")
            cmd = [
                psql,
                '-h', self.db_params['host'],
                '-p', self.db_params['port'],
                '-U', self.db_params['user'],
                '-d', db_name,
                '-c', "SELECT extname, extversion FROM pg_extension;"
            ]
            subprocess.run(cmd, env=env, check=True)
            
        except subprocess.CalledProcessError as e:
            print(f"Post-restore action failed: {e}")

    def post_restore_full_cluster(self):
        """Perform post-restore actions for all databases."""
        print("\nRunning post-restore maintenance for all databases...")
        
        env = os.environ.copy()
        env['PGPASSWORD'] = self.db_params['password']
        psql = self.get_full_path('psql')
        
        try:
            cmd = [
                psql,
                '-h', self.db_params['host'],
                '-p', self.db_params['port'],
                '-U', self.db_params['user'],
                '-d', 'postgres',
                '-c', "SELECT datname FROM pg_database WHERE datistemplate = false;",
                '-t'
            ]
            
            result = subprocess.run(cmd, env=env, 
                                  stdout=subprocess.PIPE, 
                                  stderr=subprocess.PIPE,
                                  text=True,
                                  check=True)
            
            databases = [db.strip() for db in result.stdout.splitlines() if db.strip()]
            
            for db in databases:
                if db not in ['postgres', 'template0', 'template1']:
                    print(f"\nMaintaining database: {db}")
                    self.post_restore_actions(db)
                    
        except subprocess.CalledProcessError as e:
            print(f"Failed to run full cluster post-restore actions: {e}")

    def main_menu(self):
        """Display main menu and handle user choices."""
        while True:
            print("\n" + "="*50)
            print(" POSTGRESQL MIGRATION TOOL")
            print("="*50)
            if self.target_version:
                print(f" Target Server Version: {self.target_version}")
            if hasattr(self, 'source_version') and self.source_version:
                print(f" Last Backup Version: {self.source_version}")
            print("="*50)
            
            print("\n1. Configure connection parameters")
            print("2. Create backup")
            print("3. Restore backup")
            print("4. Exit\n")
            
            choice = input("Enter your choice: ").strip()
            
            if choice == '1':
                self.configure_parameters()
            elif choice == '2':
                self.create_backup()
            elif choice == '3':
                self.restore_backup()
            elif choice == '4':
                print("Exiting...")
                break
            else:
                print("Invalid choice. Please try again.")

if __name__ == '__main__':
    migrator = PostgreSQLMigrator()
    
    if migrator.pg_bin_path is None and not migrator.is_command_available('psql'):
        print("Error: PostgreSQL binaries not found. Please ensure PostgreSQL is installed and in your PATH.")
        sys.exit(1)
        
    migrator.main_menu()