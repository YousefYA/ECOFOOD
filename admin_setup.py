import sqlite3
import hashlib
from datetime import datetime

def hash_password(password):
    """Hash password using SHA256"""
    return hashlib.sha256(password.encode()).hexdigest()

def create_admin_account():
    """Create admin account in the database"""
    
    # Database connection
    conn = sqlite3.connect('ecofood.db')
    c = conn.cursor()
    
    try:
        # Admin credentials
        username = "admin"
        email = "admin@ecofood.com"
        password = "admin123"
        hashed_password = hash_password(password)
        is_admin = True
        
        # Check if admin already exists
        c.execute("SELECT id FROM users WHERE username = ? OR email = ?", (username, email))
        existing_admin = c.fetchone()
        
        if existing_admin:
            print("❌ Admin account already exists!")
            
            # Ask if user wants to update password
            update = input("Do you want to update the admin password? (y/n): ").lower()
            if update == 'y':
                c.execute("UPDATE users SET password = ? WHERE username = ?", 
                         (hashed_password, username))
                conn.commit()
                print("✅ Admin password updated successfully!")
            else:
                print("ℹ️ No changes made.")
        else:
            # Create new admin account
            c.execute("""INSERT INTO users (username, email, password, is_admin, created_at)
                         VALUES (?, ?, ?, ?, ?)""",
                     (username, email, hashed_password, is_admin, datetime.now()))
            
            conn.commit()
            print("✅ Admin account created successfully!")
            print(f"   Username: {username}")
            print(f"   Email: {email}")
            print(f"   Password: {password}")
            print(f"   Admin privileges: {is_admin}")
    
    except sqlite3.IntegrityError as e:
        print(f"❌ Database error: {e}")
    except Exception as e:
        print(f"❌ Error creating admin account: {e}")
    finally:
        conn.close()

def create_test_user():
    """Create a test regular user account"""
    
    conn = sqlite3.connect('ecofood.db')
    c = conn.cursor()
    
    try:
        # Test user credentials
        username = "testuser"
        email = "test@ecofood.com"
        password = "test123"
        hashed_password = hash_password(password)
        is_admin = False
        
        # Check if user already exists
        c.execute("SELECT id FROM users WHERE username = ? OR email = ?", (username, email))
        existing_user = c.fetchone()
        
        if existing_user:
            print("ℹ️ Test user already exists!")
        else:
            # Create new test user
            c.execute("""INSERT INTO users (username, email, password, is_admin, created_at)
                         VALUES (?, ?, ?, ?, ?)""",
                     (username, email, hashed_password, is_admin, datetime.now()))
            
            conn.commit()
            print("✅ Test user account created successfully!")
            print(f"   Username: {username}")
            print(f"   Email: {email}")
            print(f"   Password: {password}")
            print(f"   Admin privileges: {is_admin}")
    
    except sqlite3.IntegrityError as e:
        print(f"❌ Database error: {e}")
    except Exception as e:
        print(f"❌ Error creating test user: {e}")
    finally:
        conn.close()

def initialize_database():
    """Initialize the database with required tables"""
    
    conn = sqlite3.connect('ecofood.db')
    c = conn.cursor()
    
    try:
        # Users table
        c.execute('''CREATE TABLE IF NOT EXISTS users
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      username TEXT UNIQUE NOT NULL,
                      email TEXT UNIQUE NOT NULL,
                      password TEXT NOT NULL,
                      is_admin BOOLEAN DEFAULT FALSE,
                      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                      profile_data TEXT)''')
        
        # Recipes table
        c.execute('''CREATE TABLE IF NOT EXISTS recipes
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      title TEXT NOT NULL,
                      description TEXT,
                      ingredients TEXT,
                      instructions TEXT,
                      user_id INTEGER,
                      image_data TEXT,
                      eco_score INTEGER,
                      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                      approved BOOLEAN DEFAULT FALSE,
                      FOREIGN KEY (user_id) REFERENCES users (id))''')
        
        # Food detections table
        c.execute('''CREATE TABLE IF NOT EXISTS food_detections
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      user_id INTEGER,
                      detected_items TEXT,
                      image_data TEXT,
                      confidence_scores TEXT,
                      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                      FOREIGN KEY (user_id) REFERENCES users (id))''')
        
        # Donations table
        c.execute('''CREATE TABLE IF NOT EXISTS donations
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      user_id INTEGER,
                      amount REAL,
                      recipient TEXT,
                      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                      FOREIGN KEY (user_id) REFERENCES users (id))''')
        
        # Activity logs
        c.execute('''CREATE TABLE IF NOT EXISTS activity_logs
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      user_id INTEGER,
                      action TEXT,
                      details TEXT,
                      timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                      FOREIGN KEY (user_id) REFERENCES users (id))''')
        
        conn.commit()
        print("✅ Database tables created/verified successfully!")
        
    except Exception as e:
        print(f"❌ Error initializing database: {e}")
    finally:
        conn.close()

def show_all_users():
    """Display all users in the database"""
    
    conn = sqlite3.connect('ecofood.db')
    c = conn.cursor()
    
    try:
        c.execute("SELECT id, username, email, is_admin, created_at FROM users")
        users = c.fetchall()
        
        if users:
            print("\n📊 Current Users in Database:")
            print("-" * 80)
            print(f"{'ID':<5} {'Username':<15} {'Email':<25} {'Admin':<8} {'Created':<20}")
            print("-" * 80)
            
            for user in users:
                admin_status = "Yes" if user[3] else "No"
                created_date = user[4][:19] if user[4] else "N/A"
                print(f"{user[0]:<5} {user[1]:<15} {user[2]:<25} {admin_status:<8} {created_date:<20}")
            print("-" * 80)
        else:
            print("ℹ️ No users found in database.")
            
    except Exception as e:
        print(f"❌ Error showing users: {e}")
    finally:
        conn.close()

def main():
    """Main function to run the setup script"""
    
    print("🌱 EcoFood Database Setup Script")
    print("=" * 40)
    
    # Initialize database
    print("\n1. Initializing database...")
    initialize_database()
    
    # Create admin account
    print("\n2. Creating admin account...")
    create_admin_account()
    
    # Ask if user wants to create test user
    create_test = input("\n3. Do you want to create a test user account? (y/n): ").lower()
    if create_test == 'y':
        create_test_user()
    
    # Show all users
    print("\n4. Current database users:")
    show_all_users()
    
    print("\n🎉 Setup completed!")
    print("\nYou can now login to your EcoFood app with:")
    print("   Admin: admin / admin123")
    if create_test == 'y':
        print("   Test User: testuser / test123")

if __name__ == "__main__":
    main()