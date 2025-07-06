from profile import get_profile, update_profile
import base64
import streamlit as st
import sqlite3
import hashlib
import pandas as pd
from datetime import datetime, timedelta
import json
from PIL import Image
import io
import base64

# Page configuration
st.set_page_config(
    page_title="EcoFood - Sustainable Recipe Platform",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Database initialization
def init_database():
    conn = sqlite3.connect('ecofood.db')
    c = conn.cursor()
    
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
    conn.close()

# Authentication functions
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password, hashed):
    return hash_password(password) == hashed

def register_user(username, email, password):
    conn = sqlite3.connect('ecofood.db')
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
                 (username, email, hash_password(password)))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def login_user(username, password):
    conn = sqlite3.connect('ecofood.db')
    c = conn.cursor()
    c.execute("SELECT id, username, email, is_admin, password FROM users WHERE username = ?", (username,))
    user = c.fetchone()
    conn.close()
    
    if user and verify_password(password, user[4]):
        return {
            'id': user[0],
            'username': user[1],
            'email': user[2],
            'is_admin': user[3]
        }
    return None

def log_activity(user_id, action, details=""):
    conn = sqlite3.connect('ecofood.db')
    c = conn.cursor()
    c.execute("INSERT INTO activity_logs (user_id, action, details) VALUES (?, ?, ?)",
             (user_id, action, details))
    conn.commit()
    conn.close()

# Recipe functions
def add_recipe(title, description, ingredients, instructions, user_id, image_data=None, eco_score=5):
    conn = sqlite3.connect('ecofood.db')
    c = conn.cursor()
    c.execute("""INSERT INTO recipes (title, description, ingredients, instructions, user_id, image_data, eco_score)
                 VALUES (?, ?, ?, ?, ?, ?, ?)""",
             (title, description, ingredients, instructions, user_id, image_data, eco_score))
    conn.commit()
    conn.close()
    log_activity(user_id, "Added Recipe", f"Recipe: {title}")

def get_recipes(approved_only=True, user_id=None):
    conn = sqlite3.connect('ecofood.db')
    query = """SELECT r.*, u.username FROM recipes r 
               JOIN users u ON r.user_id = u.id"""
    params = []
    
    if approved_only:
        query += " WHERE r.approved = ?"
        params.append(True)
    
    if user_id:
        if params:
            query += " AND r.user_id = ?"
        else:
            query += " WHERE r.user_id = ?"
        params.append(user_id)
    
    query += " ORDER BY r.created_at DESC"
    
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df

def search_recipes(search_term):
    conn = sqlite3.connect('ecofood.db')
    query = """SELECT r.*, u.username FROM recipes r 
               JOIN users u ON r.user_id = u.id
               WHERE r.approved = ? AND (r.title LIKE ? OR r.description LIKE ? OR r.ingredients LIKE ?)
               ORDER BY r.created_at DESC"""
    search_pattern = f"%{search_term}%"
    df = pd.read_sql_query(query, conn, params=[True, search_pattern, search_pattern, search_pattern])
    conn.close()
    return df

# Initialize database
init_database()

# Initialize session state
if 'user' not in st.session_state:
    st.session_state.user = None
if 'page' not in st.session_state:
    st.session_state.page = 'home'

# Sidebar navigation
def sidebar():
    st.sidebar.title("🌱 EcoFood Navigation")
    
    if st.session_state.user is None:
        # Not logged in
        if st.sidebar.button("🏠 Home", use_container_width=True):
            st.session_state.page = 'home'
        if st.sidebar.button("📝 Register", use_container_width=True):
            st.session_state.page = 'register'
        if st.sidebar.button("🔐 Login", use_container_width=True):
            st.session_state.page = 'login'
        if st.sidebar.button("📖 Browse Recipes", use_container_width=True):
            st.session_state.page = 'browse_recipes'
        if st.sidebar.button("🔍 Search Recipes", use_container_width=True):
            st.session_state.page = 'search_recipes'
        if st.sidebar.button("❓ Help", use_container_width=True):
            st.session_state.page = 'help'
        if st.sidebar.button("👤 Update Profile", use_container_width=True):
            st.session_state.page = "update_profile"
        if st.sidebar.button("🚪 Logout", use_container_width=True):
            st.session_state.page = "logout"

    else:
        # Logged in
        st.sidebar.write(f"Welcome, **{st.session_state.user['username']}**!")
        
        if st.sidebar.button("🏠 Home", use_container_width=True):
            st.session_state.page = 'home'
        if st.sidebar.button("👤 Update Profile", use_container_width=True):
            st.session_state.page = 'update_profile'
        if st.sidebar.button("➕ Add Recipe", use_container_width=True):
            st.session_state.page = 'add_recipe'
        if st.sidebar.button("🍽️ My Recipes", use_container_width=True):
            st.session_state.page = 'my_recipes'
        if st.sidebar.button("📖 Browse Recipes", use_container_width=True):
            st.session_state.page = 'browse_recipes'
        if st.sidebar.button("🔍 Search Recipes", use_container_width=True):
            st.session_state.page = 'search_recipes'
        if st.sidebar.button("💚 Donate", use_container_width=True):
            st.session_state.page = 'donate'
        if st.sidebar.button("📞 Contact Support", use_container_width=True):
            st.session_state.page = 'contact_support'
        
        # Admin functions
        if st.session_state.user['is_admin']:
            st.sidebar.markdown("---")
            st.sidebar.markdown("**Admin Functions**")
            if st.sidebar.button("👥 Manage Users", use_container_width=True):
                st.session_state.page = 'manage_users'
            if st.sidebar.button("📊 Monitor Activity", use_container_width=True):
                st.session_state.page = 'monitor_activity'
            if st.sidebar.button("✅ Approve Recipes", use_container_width=True):
                st.session_state.page = 'approve_recipes'
            if st.sidebar.button("📈 Analytics", use_container_width=True):
                st.session_state.page = 'analytics'
            if st.sidebar.button("⚙️ Configure Settings", use_container_width=True):
                st.session_state.page = 'configure_settings'
        
        st.sidebar.markdown("---")
        if st.sidebar.button("🚪 Logout", use_container_width=True):
            st.session_state.user = None
            st.session_state.page = 'home'
            st.rerun()

# Page functions
def home_page():
    st.title("🌱 Welcome to EcoFood")
    st.markdown("### *Sustainable Recipes for a Better Planet*")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🍽️ Total Recipes", len(get_recipes()))
    with col2:
        conn = sqlite3.connect('ecofood.db')
        user_count = pd.read_sql_query("SELECT COUNT(*) as count FROM users", conn).iloc[0]['count']
        conn.close()
        st.metric("👥 Community Members", user_count)
    with col3:
        st.metric("🌍 Eco Score Average", "8.2/10")
    
    st.markdown("---")
    st.markdown("""
    **EcoFood** is your platform for discovering and sharing sustainable, eco-friendly recipes. 
    Join our community to reduce food waste, support local ingredients, and make a positive impact on the environment!
    
    🌟 **Features:**
    - Share your sustainable recipes
    - Discover eco-friendly cooking tips
    - Track your environmental impact
    - Connect with like-minded food enthusiasts
    - Support food donation initiatives
    """)

def register_page():
    st.title("📝 Register for EcoFood")
    
    with st.form("register_form"):
        username = st.text_input("Username")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        
        if st.form_submit_button("Register"):
            if password != confirm_password:
                st.error("Passwords don't match!")
            elif len(password) < 6:
                st.error("Password must be at least 6 characters long!")
            elif register_user(username, email, password):
                st.success("Registration successful! Please login.")
                st.session_state.page = 'login'
                st.rerun()
            else:
                st.error("Username or email already exists!")

def login_page():
    st.title("🔐 Login to EcoFood")
    
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        if st.form_submit_button("Login"):
            user = login_user(username, password)
            if user:
                st.session_state.user = user
                log_activity(user['id'], "Login")
                st.success("Login successful!")
                st.session_state.page = 'home'
                st.rerun()
            else:
                st.error("Invalid username or password!")

def add_recipe_page():
    st.title("➕ Add New Recipe")
    
    with st.form("recipe_form"):
        title = st.text_input("Recipe Title")
        description = st.text_area("Description")
        ingredients = st.text_area("Ingredients (one per line)")
        instructions = st.text_area("Instructions")
        
        eco_score = st.slider("Eco-Friendliness Score", 1, 10, 5,
                             help="Rate how environmentally friendly this recipe is")
        
        uploaded_file = st.file_uploader("Recipe Image", type=['png', 'jpg', 'jpeg'])
        
        if st.form_submit_button("Add Recipe"):
            if title and ingredients and instructions:
                image_data = None
                if uploaded_file:
                    image_data = base64.b64encode(uploaded_file.read()).decode()
                
                add_recipe(title, description, ingredients, instructions, 
                          st.session_state.user['id'], image_data, eco_score)
                st.success("Recipe added successfully! It will be reviewed before publication.")
                st.session_state.page = 'my_recipes'
                st.rerun()
            else:
                st.error("Please fill in all required fields!")

def browse_recipes_page():
    st.title("📖 Browse Recipes")
    
    recipes_df = get_recipes()
    
    if len(recipes_df) > 0:
        for _, recipe in recipes_df.iterrows():
            with st.container():
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    if recipe['image_data']:
                        try:
                            image_data = base64.b64decode(recipe['image_data'])
                            image = Image.open(io.BytesIO(image_data))
                            st.image(image, width=200)
                        except:
                            st.write("📷 Image")
                    else:
                        st.write("📷 No Image")
                
                with col2:
                    st.subheader(recipe['title'])
                    st.write(f"*By {recipe['username']}*")
                    st.write(recipe['description'])
                    
                    col2_1, col2_2 = st.columns(2)
                    with col2_1:
                        st.write(f"🌍 Eco Score: {recipe['eco_score']}/10")
                    with col2_2:
                        st.write(f"📅 {recipe['created_at']}")
                    
                    if st.button(f"View Full Recipe", key=f"view_{recipe['id']}"):
                        st.session_state.selected_recipe = recipe['id']
                        st.session_state.page = 'view_recipe'
                        st.rerun()
                
                st.markdown("---")
    else:
        st.info("No recipes available yet. Be the first to add one!")

def search_recipes_page():
    st.title("🔍 Search Recipes")
    
    search_term = st.text_input("Search for recipes...")
    
    if search_term:
        results = search_recipes(search_term)
        
        if len(results) > 0:
            st.write(f"Found {len(results)} recipe(s)")
            
            for _, recipe in results.iterrows():
                with st.container():
                    st.subheader(recipe['title'])
                    st.write(f"*By {recipe['username']}*")
                    st.write(recipe['description'])
                    st.write(f"🌍 Eco Score: {recipe['eco_score']}/10")
                    st.markdown("---")
        else:
            st.info("No recipes found matching your search.")

def manage_users_page():
    if not st.session_state.user['is_admin']:
        st.error("Access denied. Admin privileges required.")
        return
    
    st.title(" Manage Users")
    
    conn = sqlite3.connect('ecofood.db')
    users_df = pd.read_sql_query("SELECT id, username, email, is_admin, created_at FROM users", conn)
    conn.close()
    
    st.dataframe(users_df, use_container_width=True)


def update_profile_page():
    user = st.session_state.get("user")
    if not user:
        st.warning("🔒 You must log in before updating your profile.")
        st.session_state.page = "login"
        st.experimental_rerun()
        return

    st.title("👤 Update Your Profile")
    user_id = user["id"]
    profile = get_profile(user_id) or {}

    with st.form("profile_form"):
        name     = st.text_input("Full Name",    profile.get("name", ""))
        bio      = st.text_area("Bio",           profile.get("bio", ""))
        location = st.text_input("Location",     profile.get("location", ""))
        avatar   = st.file_uploader("Avatar",     type=["png","jpg","jpeg"])

        if st.form_submit_button("Save"):
            profile["name"]     = name
            profile["bio"]      = bio
            profile["location"] = location
            if avatar:
                profile["avatar"] = base64.b64encode(avatar.read()).decode()
            update_profile(user_id, profile)
            st.success("✅ Profile updated!")
            st.session_state.page = "home"
            st.experimental_rerun()

def donate_page():
    st.title("💚 Support Food Donations")
    
    st.markdown("""
    Help us support local food banks and sustainable food initiatives!
    """)
    
    with st.form("donation_form"):
        amount = st.number_input("Donation Amount ($)", min_value=1.0, step=1.0)
        recipient = st.selectbox("Choose Recipient", 
                                ["Local Food Bank", "Community Garden", "Sustainable Farming Initiative"])
        
        if st.form_submit_button("Donate"):
            conn = sqlite3.connect('ecofood.db')
            c = conn.cursor()
            c.execute("INSERT INTO donations (user_id, amount, recipient) VALUES (?, ?, ?)",
                     (st.session_state.user['id'], amount, recipient))
            conn.commit()
            conn.close()
            
            log_activity(st.session_state.user['id'], "Donation", f"${amount} to {recipient}")
            st.success(f"Thank you for your ${amount:.2f} donation to {recipient}!")

# Main app logic
def main():
    sidebar()
    
    # Route to appropriate page
    if st.session_state.page == 'home':
        home_page()
    elif st.session_state.page == 'register':
        register_page()
    elif st.session_state.page == 'login':
        login_page()
    elif st.session_state.page == 'add_recipe':
        if st.session_state.user:
            add_recipe_page()
        else:
            st.error("Please login to add recipes.")
    elif st.session_state.page == 'browse_recipes':
        browse_recipes_page()
    elif st.session_state.page == 'search_recipes':
        search_recipes_page()
    elif st.session_state.page == 'manage_users':
        manage_users_page()
    elif st.session_state.page == 'approve_recipes':
        approve_recipes_page()
    elif st.session_state.page == 'donate':
        if st.session_state.user:
            donate_page()
        else:
            st.error("Please login to make donations.")
    elif st.session_state.page == 'help':
        st.title("❓ Help & Support")
        st.markdown("""
        **Welcome to EcoFood Help Center!**
        
        **Getting Started:**
        - Register for a free account
        - Add your first sustainable recipe
        - Browse recipes from our community
        
        **Features:**
        - Recipe sharing with eco-scoring
        - Community donations
        - Recipe search and filtering
        - User profiles and activity tracking
        
        **Need More Help?**
        Contact our support team through the Contact Support page.
        """)
    elif st.session_state.page == 'contact_support':
        st.title("📞 Contact Support")
        with st.form("support_form"):
            subject = st.text_input("Subject")
            message = st.text_area("Message")
            if st.form_submit_button("Send Message"):
                st.success("Your message has been sent! We'll get back to you soon.")
    elif st.session_state.page == "update_profile":
        update_profile_page()
    elif st.session_state.page == "logout":
        logout_page()
    else:
        # Default to home
        home_page()

if __name__ == "__main__":
    main()