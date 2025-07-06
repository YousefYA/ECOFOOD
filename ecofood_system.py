import streamlit as st
import sqlite3
import hashlib
import pandas as pd
from datetime import datetime, timedelta
import json
from PIL import Image
import io
import base64
import cv2
import numpy as np
from ultralytics import YOLO
import openai
import os
from typing import List, Dict
import torch
from profile import get_profile, update_profile
import base64

# Page configuration
st.set_page_config(
    page_title="EcoFood - Sustainable Recipe Platform",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for food detection
if 'detected_foods' not in st.session_state:
    st.session_state.detected_foods = []
if 'recommended_recipe' not in st.session_state:
    st.session_state.recommended_recipe = None

# Food Detection Class
class FoodDetector:
    def __init__(self):
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load YOLOv8 model for food detection"""
        try:
            # Try to load a custom food detection model first
            if os.path.exists('food_detection_yolo.pt'):
                self.model = YOLO('food_detection_yolo.pt')
                st.success("✅ Custom food detection model loaded!")
            else:
                # Fallback to general YOLOv8 model
                self.model = YOLO('yolov8n.pt')  # Nano version for faster inference
                st.info("ℹ️ Using general YOLOv8 model (for demo purposes)")
        except Exception as e:
            st.error(f"❌ Error loading YOLO model: {str(e)}")
            self.model = None
    
    def detect_food_items(self, image):
        """Detect food items in the image"""
        if self.model is None:
            return self.mock_detection(image)
        
        try:
            # Convert PIL image to numpy array
            img_array = np.array(image)
            
            # Run YOLO detection
            results = self.model(img_array)
            
            detected_foods = []
            food_related_classes = [
                'apple', 'banana', 'orange', 'broccoli', 'carrot', 'pizza', 
                'donut', 'cake', 'chair', 'dining table', 'cup', 'fork', 
                'knife', 'spoon', 'bowl', 'bottle', 'wine glass', 'sandwich',
                'hot dog', 'person'  # We'll filter these
            ]
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        class_name = self.model.names[class_id]
                        
                        # Filter for food-related items with high confidence
                        if confidence > 0.5 and self.is_food_related(class_name):
                            detected_foods.append({
                                'name': class_name,
                                'confidence': confidence,
                                'bbox': box.xyxy[0].tolist()  # Bounding box coordinates
                            })
            
            return detected_foods
        
        except Exception as e:
            st.error(f"Detection error: {str(e)}")
            return self.mock_detection(image)
    
    def is_food_related(self, class_name):
        """Check if detected class is food-related"""
        food_keywords = [
            'apple', 'banana', 'orange', 'broccoli', 'carrot', 'pizza', 
            'donut', 'cake', 'sandwich', 'hot dog', 'cup', 'bowl', 
            'bottle', 'wine glass', 'fork', 'knife', 'spoon'
        ]
        return any(keyword in class_name.lower() for keyword in food_keywords)
    
    def mock_detection(self, image):
        """Mock detection for demo purposes when YOLO isn't available"""
        # Simulate food detection based on image characteristics
        mock_foods = [
            {'name': 'tomato', 'confidence': 0.85},
            {'name': 'onion', 'confidence': 0.78},
            {'name': 'garlic', 'confidence': 0.72},
            {'name': 'bell pepper', 'confidence': 0.69}
        ]
        
        # Return 1-3 random items for demo
        import random
        num_items = random.randint(1, 3)
        return random.sample(mock_foods, num_items)

# OpenAI Recipe Recommender
class RecipeRecommender:
    def __init__(self, api_key=None):
        if api_key:
            openai.api_key = api_key
        else:
            # For demo purposes, we'll use a mock recommender
            self.use_mock = True
    
    def recommend_recipe(self, ingredients: List[str]) -> Dict:
        """Get recipe recommendation from OpenAI based on detected ingredients"""
        if hasattr(self, 'use_mock'):
            return self.mock_recommendation(ingredients)
        
        try:
            # Create prompt for OpenAI
            ingredients_text = ", ".join(ingredients)
            prompt = f"""
            Create a sustainable, eco-friendly recipe using these ingredients: {ingredients_text}
            
            Please provide:
            1. Recipe title
            2. Brief description focusing on sustainability
            3. Complete ingredients list (including the detected items)
            4. Step-by-step instructions
            5. Eco-friendliness tips
            6. Estimated eco-score (1-10, where 10 is most eco-friendly)
            
            Format as JSON with keys: title, description, ingredients, instructions, eco_tips, eco_score
            """
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert sustainable cooking assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.7
            )
            
            # Parse response
            recipe_text = response.choices[0].message.content
            recipe_data = json.loads(recipe_text)
            
            return recipe_data
        
        except Exception as e:
            st.error(f"Error getting recipe recommendation: {str(e)}")
            return self.mock_recommendation(ingredients)
    
    def mock_recommendation(self, ingredients: List[str]) -> Dict:
        """Mock recipe recommendation for demo purposes"""
        ingredients_text = ", ".join(ingredients)
        
        mock_recipes = {
            "sustainable_stir_fry": {
                "title": f"Sustainable {ingredients_text.title()} Stir-Fry",
                "description": "A quick and eco-friendly stir-fry that makes the most of your fresh ingredients while minimizing food waste.",
                "ingredients": [
                    f"2 cups {ingredients[0]} (detected)",
                    f"1 cup {ingredients[1] if len(ingredients) > 1 else 'mixed vegetables'} (detected)",
                    "2 tbsp olive oil (locally sourced)",
                    "2 cloves garlic, minced",
                    "1 tbsp soy sauce (low sodium)",
                    "1 tsp sesame oil",
                    "Fresh herbs (basil or cilantro)",
                    "Optional: brown rice or quinoa"
                ],
                "instructions": [
                    "Heat olive oil in a large pan or wok over medium-high heat",
                    f"Add {ingredients[0]} and cook for 3-4 minutes until tender",
                    "Add garlic and other detected vegetables, stir-fry for 2-3 minutes",
                    "Add soy sauce and sesame oil, toss to combine",
                    "Cook for another 1-2 minutes until vegetables are crisp-tender",
                    "Garnish with fresh herbs and serve over rice or quinoa",
                    "Save any leftovers for tomorrow's lunch to reduce waste!"
                ],
                "eco_tips": [
                    "Use locally sourced vegetables when possible",
                    "Compost any vegetable scraps",
                    "Choose organic ingredients to support sustainable farming",
                    "Save cooking water for plants after it cools"
                ],
                "eco_score": 8
            }
        }
        
        return mock_recipes["sustainable_stir_fry"]

# Initialize food detector and recipe recommender
@st.cache_resource
def load_food_detector():
    return FoodDetector()

@st.cache_resource
def load_recipe_recommender():
    # Try to get OpenAI API key from environment or user input
    api_key = os.getenv('OPENAI_API_KEY')
    return RecipeRecommender("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")

# Database initialization (keeping previous functions)
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
    
    # Food detections table (new)
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
    conn.close()

# Authentication functions (keeping previous functions)
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

# Food detection storage functions
def save_food_detection(user_id, detected_items, image_data=None, confidence_scores=None):
    conn = sqlite3.connect('ecofood.db')
    c = conn.cursor()
    c.execute("""INSERT INTO food_detections (user_id, detected_items, image_data, confidence_scores)
                 VALUES (?, ?, ?, ?)""",
             (user_id, json.dumps(detected_items), image_data, json.dumps(confidence_scores)))
    conn.commit()
    conn.close()

# Recipe functions (keeping previous functions)
def add_recipe(title, description, ingredients, instructions, user_id, image_data=None, eco_score=5):
    conn = sqlite3.connect('ecofood.db')
    c = conn.cursor()
    c.execute("""INSERT INTO recipes (title, description, ingredients, instructions, user_id, image_data, eco_score)
                 VALUES (?, ?, ?, ?, ?, ?, ?)""",
             (title, description, ingredients, instructions, user_id, image_data, eco_score))
    conn.commit()
    conn.close()
    log_activity(user_id, "Added Recipe", f"Recipe: {title}")

def get_recipes(approved_only=False, user_id=None):
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
    
    try:
        df = pd.read_sql_query(query, conn, params=params)
    except:
        df = pd.DataFrame()
    finally:
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

# OpenAI API key setup
openai_api_key = "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"
    
# Food Detection Page
def food_detection_page():
    st.title("📷 Smart Food Detection")
    st.markdown("*Scan your ingredients and get sustainable recipe recommendations!*")
    
    # OpenAI API key setup
    if not openai_api_key:
        st.warning("⚠️ Please enter your OpenAI API key in the sidebar to use food detection.")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🔍 Image Input")
        
        # Camera input
        camera_image = st.camera_input("📸 Take a picture of your ingredients")
        
        # File upload alternative - allow multiple files
        uploaded_files = st.file_uploader(
            "📁 Or upload images", 
            type=['png', 'jpg', 'jpeg'], 
            accept_multiple_files=True,
            help="You can upload multiple images to scan different ingredients"
        )
        
        # Process camera image
        if camera_image:
            image = Image.open(camera_image)
            st.image(image, caption="Camera image to analyze", width=400)
            
            if st.button("🔍 Detect from Camera", type="primary"):
                with st.spinner("🤖 Analyzing camera image..."):
                    detect_and_add_ingredients(camera_image, openai_api_key)
        
        # Process uploaded files
        if uploaded_files:
            st.write(f"📎 {len(uploaded_files)} image(s) uploaded")
            
            # Show thumbnails of uploaded images
            cols = st.columns(min(len(uploaded_files), 3))
            for idx, uploaded_file in enumerate(uploaded_files):
                with cols[idx % 3]:
                    image = Image.open(uploaded_file)
                    st.image(image, caption=f"Image {idx+1}", width=120)
            
            # Detect from all uploaded images
            if st.button("🔍 Detect from All Images", type="primary"):
                with st.spinner(f"🤖 Analyzing {len(uploaded_files)} image(s)..."):
                    for idx, uploaded_file in enumerate(uploaded_files):
                        st.info(f"Processing image {idx+1}/{len(uploaded_files)}...")
                        detect_and_add_ingredients(uploaded_file, openai_api_key, show_success=False)
                    
                    st.success(f"✅ Processed all {len(uploaded_files)} images!")
                    st.rerun()
    
    with col2:
        st.subheader("🥗 Ingredient Collection")
        
        # Show total count
        if st.session_state.detected_foods:
            st.info(f"📊 Total ingredients: {len(st.session_state.detected_foods)}")
        
        if st.session_state.detected_foods:
            # Display detected foods with individual remove buttons
            st.markdown("**Collected ingredients:**")
            
            # Edit mode toggle
            edit_mode = st.checkbox("✏️ Edit ingredients", key="edit_ingredients")
            
            if edit_mode:
                # Editable list with individual remove buttons
                ingredients_to_keep = []
                
                for i, food in enumerate(st.session_state.detected_foods):
                    col_name, col_remove = st.columns([3, 1])
                    
                    with col_name:
                        edited_name = st.text_input(
                            f"Item {i+1}:", 
                            value=food['name'], 
                            key=f"edit_food_{i}"
                        )
                    
                    with col_remove:
                        if st.button("🗑️", key=f"remove_{i}", help="Remove this ingredient"):
                            # Skip this ingredient (don't add to keep list)
                            continue
                    
                    if edited_name.strip():
                        ingredients_to_keep.append({
                            'name': edited_name.strip(),
                            'confidence': food.get('confidence', 0.8)
                        })
                
                # Add new ingredient option
                new_ingredient = st.text_input("➕ Add ingredient manually:", key="new_ingredient")
                if new_ingredient.strip():
                    ingredients_to_keep.append({
                        'name': new_ingredient.strip(),
                        'confidence': 1.0
                    })
                
                # Update button
                if st.button("💾 Update Ingredients"):
                    st.session_state.detected_foods = ingredients_to_keep
                    st.success("✅ Ingredients updated!")
                    st.rerun()
            else:
                # Display only mode with remove buttons
                ingredients_to_remove = []
                
                for i, food in enumerate(st.session_state.detected_foods):
                    col_name, col_remove = st.columns([4, 1])
                    
                    with col_name:
                        st.write(f"🥬 **{food['name'].title()}**")
                    
                    with col_remove:
                        if st.button("❌", key=f"remove_display_{i}", help="Remove this ingredient"):
                            ingredients_to_remove.append(i)
                
                # Remove selected ingredients
                if ingredients_to_remove:
                    for idx in sorted(ingredients_to_remove, reverse=True):
                        st.session_state.detected_foods.pop(idx)
                    st.rerun()
            
            # Manual add ingredient (always visible)
            st.markdown("---")
            manual_ingredient = st.text_input("🖊️ Add ingredient manually:", key="manual_add")
            if st.button("➕ Add") and manual_ingredient.strip():
                # Check if ingredient already exists
                existing_names = [food['name'].lower() for food in st.session_state.detected_foods]
                if manual_ingredient.lower() not in existing_names:
                    st.session_state.detected_foods.append({
                        'name': manual_ingredient.strip(),
                        'confidence': 1.0
                    })
                    st.success(f"✅ Added {manual_ingredient}")
                    st.rerun()
                else:
                    st.warning("⚠️ Ingredient already exists!")
            
            # Control buttons
            st.markdown("---")
            col2_1, col2_2 = st.columns(2)
            
            with col2_1:
                if st.button("🍽️ Get Recipe", type="primary", use_container_width=True):
                    with st.spinner("🤖 Creating recipe recommendation..."):
                        ingredients = [food['name'] for food in st.session_state.detected_foods]
                        
                        try:
                            # OpenAI recipe recommendation
                            client = openai.OpenAI(api_key=openai_api_key)
                            
                            ingredients_text = ", ".join(ingredients)
                            prompt = f"""Create a sustainable recipe that MUST include ALL of these ingredients: {ingredients_text}

Please provide a JSON response with:
- title: Recipe name
- description: Brief eco-friendly description
- ingredients: Complete list (including ALL detected ingredients)
- instructions: Step-by-step cooking instructions
- eco_tips: 3 sustainability tips
- eco_score: Score from 1-10 for eco-friendliness

The recipe must use every single ingredient I mentioned: {ingredients_text}"""

                            response = client.chat.completions.create(
                                model="gpt-4o-mini",
                                messages=[
                                    {"role": "system", "content": "You are an expert sustainable cooking assistant. Always create recipes that use ALL provided ingredients."},
                                    {"role": "user", "content": prompt}
                                ],
                                max_tokens=1000,
                                temperature=0.7
                            )
                            
                            # Parse JSON response
                            recipe_text = response.choices[0].message.content
                            
                            # Clean up JSON if wrapped in markdown
                            if "```json" in recipe_text:
                                recipe_text = recipe_text.split("```json")[1].split("```")[0]
                            elif "```" in recipe_text:
                                recipe_text = recipe_text.split("```")[1]
                            
                            recipe = json.loads(recipe_text)
                            st.session_state.recommended_recipe = recipe
                            
                            # Log activity
                            if st.session_state.user:
                                log_activity(st.session_state.user['id'], "Recipe Recommendation", f"For: {', '.join(ingredients)}")
                            
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"❌ Error generating recipe: {str(e)}")
            
            with col2_2:
                if st.button("🗑️ Reset All", use_container_width=True):
                    st.session_state.detected_foods = []
                    st.session_state.recommended_recipe = None
                    st.success("✅ All ingredients cleared!")
                    st.rerun()
        else:
            st.info("👆 Scan or upload images to detect ingredients!")
            st.markdown("💡 **Tip:** You can scan multiple images to collect more ingredients!")
    
    # Display recommended recipe (same as before)
    if st.session_state.recommended_recipe:
        st.markdown("---")
        st.subheader("🌟 Recommended Recipe")
        
        recipe = st.session_state.recommended_recipe
        
        # Recipe header
        col_title, col_score = st.columns([3, 1])
        with col_title:
            st.markdown(f"### {recipe['title']}")
        with col_score:
            st.metric("🌍 Eco Score", f"{recipe['eco_score']}/10")
        
        st.markdown(recipe['description'])
        
        # Recipe details in tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📝 Ingredients", "👨‍🍳 Instructions", "🌱 Eco Tips", "💾 Actions"])
        
        with tab1:
            st.markdown("**Ingredients:**")
            for ingredient in recipe['ingredients']:
                st.write(f"• {ingredient}")
        
        with tab2:
            st.markdown("**Instructions:**")
            for i, instruction in enumerate(recipe['instructions'], 1):
                st.write(f"{i}. {instruction}")
        
        with tab3:
            st.markdown("**Eco-Friendly Tips:**")
            for tip in recipe['eco_tips']:
                st.write(f"🌱 {tip}")
        
        with tab4:
            if st.session_state.user:
                if st.button("💾 Save Recipe to My Collection"):
                    # Save the recommended recipe to user's collection
                    ingredients_text = "\n".join(recipe['ingredients'])
                    instructions_text = "\n".join(recipe['instructions'])
                    
                    add_recipe(
                        title=recipe['title'],
                        description=recipe['description'],
                        ingredients=ingredients_text,
                        instructions=instructions_text,
                        user_id=st.session_state.user['id'],
                        eco_score=recipe['eco_score']
                    )
                    
                    st.success("✅ Recipe saved to your collection!")
            else:
                st.info("🔐 Login to save recipes to your collection")
            
            if st.button("📤 Share Recipe"):
                recipe_text = f"""
**{recipe['title']}**

{recipe['description']}

**Ingredients:**
{chr(10).join(['• ' + ing for ing in recipe['ingredients']])}

**Instructions:**
{chr(10).join([f'{i+1}. {inst}' for i, inst in enumerate(recipe['instructions'])])}

**Eco Score:** {recipe['eco_score']}/10
                """
                st.text_area("📋 Copy this recipe text:", recipe_text, height=200)

# Helper function for detecting and adding ingredients
def detect_and_add_ingredients(image_file, openai_api_key, show_success=True):
    """Detect ingredients from image and add to session state"""
    try:
        # Convert image to base64
        image = Image.open(image_file)
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        # OpenAI Vision API call
        client = openai.OpenAI(api_key=openai_api_key)
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Identify all food ingredients in this image. List only the ingredient names, 1-2 words each, separated by commas. Example: tomato, onion, garlic"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_base64}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=100
        )
        
        # Parse detected ingredients
        detected_text = response.choices[0].message.content.strip()
        new_ingredients = [ing.strip() for ing in detected_text.split(",") if ing.strip()]
        
        if new_ingredients:
            # Get existing ingredient names (case insensitive)
            existing_names = [food['name'].lower() for food in st.session_state.detected_foods]
            
            # Add only new ingredients (avoid duplicates)
            added_count = 0
            for ingredient in new_ingredients:
                if ingredient.lower() not in existing_names:
                    st.session_state.detected_foods.append({
                        'name': ingredient,
                        'confidence': 0.9
                    })
                    existing_names.append(ingredient.lower())  # Update existing list
                    added_count += 1
            
            # Save detection to database if user is logged in
            if st.session_state.user and added_count > 0:
                try:
                    image_data = base64.b64encode(image_file.read()).decode()
                    confidence_scores = {ing: 0.9 for ing in new_ingredients}
                    save_food_detection(
                        st.session_state.user['id'], 
                        new_ingredients,
                        image_data,
                        confidence_scores
                    )
                    log_activity(st.session_state.user['id'], "Food Detection", f"Added: {added_count} new items")
                except:
                    pass  # Ignore database errors
            
            if show_success:
                if added_count > 0:
                    st.success(f"✅ Added {added_count} new ingredient(s)! (Skipped {len(new_ingredients) - added_count} duplicates)")
                else:
                    st.info("ℹ️ All detected ingredients already exist in your collection!")
                st.rerun()
            
            return added_count
        else:
            if show_success:
                st.warning("🤔 No food items detected in this image.")
            return 0
            
    except Exception as e:
        if show_success:
            st.error(f"❌ Error detecting food items: {str(e)}")
        return 0

# Sidebar navigation (updated to include food detection)
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
        if st.sidebar.button("📷 Food Detection", use_container_width=True):
            st.session_state.page = 'food_detection'
        if st.sidebar.button("📖 Browse Recipes", use_container_width=True):
            st.session_state.page = 'browse_recipes'
        if st.sidebar.button("🔍 Search Recipes", use_container_width=True):
            st.session_state.page = 'search_recipes'
        if st.sidebar.button("❓ Help", use_container_width=True):
            st.session_state.page = 'help'
    else:
        # Logged in
        st.sidebar.write(f"Welcome, **{st.session_state.user['username']}**!")
        
        if st.sidebar.button("🏠 Home", use_container_width=True):
            st.session_state.page = 'home'
        if st.sidebar.button("👤 Update Profile", use_container_width=True):
            st.session_state.page = 'update_profile'
        if st.sidebar.button("📷 Food Detection", use_container_width=True):
            st.session_state.page = 'food_detection'
        if st.sidebar.button("➕ Add Recipe", use_container_width=True):
            st.session_state.page = 'add_recipe'
        if st.sidebar.button("🍽️ My Recipes", use_container_width=True):
            st.session_state.page = 'my_recipes'
        if st.sidebar.button("📖 Browse Recipes", use_container_width=True):
            st.session_state.page = 'browse_recipes'
        if st.sidebar.button("🔍 Search Recipes", use_container_width=True):
            st.session_state.page = 'search_recipes'
        if st.sidebar.button("📞 Contact Support", use_container_width=True):
            st.session_state.page = 'contact_support'
        
        # Admin functions
        if st.session_state.user.get('is_admin'):
            st.sidebar.markdown("---")
            st.sidebar.markdown("**Admin Functions**")
            if st.sidebar.button("👥 Manage Users", use_container_width=True):
                st.session_state.page = 'manage_users'
            if st.sidebar.button("📊 Monitor Activity", use_container_width=True):
                st.session_state.page = 'monitor_activity'
            if st.sidebar.button("✅ Approve Recipes", use_container_width=True):
                st.session_state.page = 'approve_recipes'
        
        st.sidebar.markdown("---")
        if st.sidebar.button("🚪 Logout", use_container_width=True):
            # Clear session on logout
            st.session_state.user = None
            st.session_state.page = 'home'
            st.session_state.detected_foods = []
            st.session_state.recommended_recipe = None
            st.experimental_rerun()


def get_recipe_by_id(recipe_id):
    conn = sqlite3.connect('ecofood.db')
    c = conn.cursor()
    c.execute("""SELECT r.*, u.username FROM recipes r 
                 JOIN users u ON r.user_id = u.id 
                 WHERE r.id = ?""", (recipe_id,))
    recipe = c.fetchone()
    conn.close()
    return recipe

def view_recipe_page():
    if 'selected_recipe_id' not in st.session_state:
        st.error("❌ No recipe selected")
        if st.button("🔙 Back to Browse"):
            st.session_state.page = 'browse_recipes'
            st.rerun()
        return
    
    recipe_data = get_recipe_by_id(st.session_state.selected_recipe_id)
    
    if not recipe_data:
        st.error("❌ Recipe not found")
        if st.button("🔙 Back to Browse"):
            st.session_state.page = 'browse_recipes'
            st.rerun()
        return
    
    # Unpack recipe data
    (recipe_id, title, description, ingredients, instructions, user_id, 
     image_data, eco_score, created_at, approved, username) = recipe_data
    
    # Header with back button
    col_back, col_title = st.columns([1, 5])
    with col_back:
        if st.button("🔙 Back"):
            st.session_state.page = 'browse_recipes'
            st.rerun()
    
    with col_title:
        st.title(title)
    
    # Recipe metadata
    col_meta1, col_meta2, col_meta3 = st.columns(3)
    with col_meta1:
        st.metric("👤 Chef", username)
    with col_meta2:
        st.metric("🌍 Eco Score", f"{eco_score}/10")
    with col_meta3:
        st.metric("📅 Published", created_at[:10])
    
    # Recipe image
    if image_data:
        try:
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            st.image(image, caption=title, width=500)
        except:
            st.info("📷 Image unavailable")
    
    # Description
    if description:
        st.markdown("### 📝 Description")
        st.write(description)
    
    # Recipe content in tabs
    tab1, tab2, tab3 = st.tabs(["🥬 Ingredients", "👨‍🍳 Instructions", "💾 Actions"])
    
    with tab1:
        st.markdown("### Ingredients")
        ingredients_list = ingredients.split('\n')
        for ingredient in ingredients_list:
            if ingredient.strip():
                st.write(f"• {ingredient.strip()}")
    
    with tab2:
        st.markdown("### Instructions")
        instructions_list = instructions.split('\n')
        for i, instruction in enumerate(instructions_list, 1):
            if instruction.strip():
                st.write(f"**Step {i}:** {instruction.strip()}")
    
    with tab3:
        st.markdown("### Actions")
        
        # Save to collection (if logged in)
        if st.session_state.user:
            if st.button("💾 Save to My Collection", use_container_width=True):
                try:
                    add_recipe(
                        title=f"[Saved] {title}",
                        description=description,
                        ingredients=ingredients,
                        instructions=instructions,
                        user_id=st.session_state.user['id'],
                        image_data=image_data,
                        eco_score=eco_score
                    )
                    st.success("✅ Recipe saved to your collection!")
                except:
                    st.error("❌ Error saving recipe")
        
        # Share recipe
        if st.button("📤 Share Recipe", use_container_width=True):
            recipe_text = f"""
**{title}**
*By {username}*

**Description:**
{description}

**Ingredients:**
{chr(10).join(['• ' + ing.strip() for ing in ingredients.split(chr(10)) if ing.strip()])}

**Instructions:**
{chr(10).join([f'{i+1}. {inst.strip()}' for i, inst in enumerate(instructions.split(chr(10))) if inst.strip()])}

**Eco Score:** {eco_score}/10
            """
            st.text_area("📋 Copy this recipe:", recipe_text, height=300)
        
        # Report recipe
        if st.button("🚨 Report Recipe", use_container_width=True):
            st.warning("⚠️ Recipe reported. Thank you for keeping our community safe!")
# Page functions (keeping all previous page functions and adding food detection routing)
def home_page():
    st.title("🌱 Welcome to EcoFood")
    st.markdown("### *Sustainable Recipes for a Better Planet*")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🍽️ Total Recipes", len(get_recipes()))
    with col2:
        conn = sqlite3.connect('ecofood.db')
        user_count = pd.read_sql_query("SELECT COUNT(*) as count FROM users", conn).iloc[0]['count']
        conn.close()
        st.metric("👥 Community Members", user_count)
    with col3:
        st.metric("🌍 Eco Score Average", "8.2/10")
    with col4:
        # Count food detections
        conn = sqlite3.connect('ecofood.db')
        detection_count = pd.read_sql_query("SELECT COUNT(*) as count FROM food_detections", conn).iloc[0]['count']
        conn.close()
        st.metric("📷 Food Scans", detection_count)
    
    st.markdown("---")
    
    # Highlight food detection feature
    st.markdown("## 🆕 New Feature: Smart Food Detection")
    
    col_demo1, col_demo2 = st.columns(2)
    with col_demo1:
        st.markdown("""
        **🔍 AI-Powered Food Recognition**
        - Take a photo of your ingredients
        - Get instant AI identification
        - Receive personalized sustainable recipes
        - Edit detected items for accuracy
        """)
    
    with col_demo2:
        if st.button("🚀 Try Food Detection Now!", type="primary", use_container_width=True):
            st.session_state.page = 'food_detection'
            st.rerun()
    
    st.markdown("---")
    st.markdown("""
    **EcoFood** is your platform for discovering and sharing sustainable, eco-friendly recipes. 
    Join our community to reduce food waste, support local ingredients, and make a positive impact on the environment!
    
    🌟 **Features:**
    - 📷 **NEW**: AI-powered food detection and recipe recommendations
    - Share your sustainable recipes
    - Discover eco-friendly cooking tips
    - Track your environmental impact
    - Connect with like-minded food enthusiasts
    - Support food donation initiatives
    """)

# Keep all other page functions from before (register_page, login_page, etc.)
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


def delete_recipe(recipe_id):
    conn = sqlite3.connect('ecofood.db')
    c = conn.cursor()
    c.execute("DELETE FROM recipes WHERE id = ?", (recipe_id,))
    conn.commit()
    conn.close()
    
def my_recipes_page():
    if not st.session_state.user:
        st.error("🔐 Please login to view your recipes.")
        return
    
    st.title("🍽️ My Recipes")
    st.markdown(f"*Recipes by {st.session_state.user['username']}*")
    
    # Get user's recipes
    recipes_df = get_recipes(approved_only=False, user_id=st.session_state.user['id'])
    
    if len(recipes_df) == 0:
        st.info("📝 You haven't added any recipes yet!")
        if st.button("➕ Add Your First Recipe"):
            st.session_state.page = 'add_recipe'
            st.rerun()
        return
    
    # Stats
    approved_count = len(recipes_df[recipes_df['approved'] == True])
    pending_count = len(recipes_df[recipes_df['approved'] == False])
    avg_eco_score = recipes_df['eco_score'].mean()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("✅ Approved", approved_count)
    with col2:
        st.metric("⏳ Pending", pending_count)
    with col3:
        st.metric("🌍 Avg Eco Score", f"{avg_eco_score:.1f}/10")
    
    st.markdown("---")
    
    # Display recipes
    for _, recipe in recipes_df.iterrows():
        with st.container():
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                # Status indicator
                status = "✅ Approved" if recipe['approved'] else "⏳ Pending Review"
                st.subheader(f"{recipe['title']} - {status}")
                st.write(f"🌍 Eco Score: {recipe['eco_score']}/10")
                st.write(f"📅 Created: {recipe['created_at'][:10]}")
                
                # Show description (truncated)
                if recipe['description']:
                    desc = recipe['description']
                    if len(desc) > 150:
                        desc = desc[:150] + "..."
                    st.write(desc)
            
            with col2:
                if st.button("👀 View", key=f"view_my_{recipe['id']}", use_container_width=True):
                    st.session_state.selected_recipe_id = recipe['id']
                    st.session_state.page = 'view_recipe'
                    st.rerun()
                
                if st.button("✏️ Edit", key=f"edit_{recipe['id']}", use_container_width=True):
                    st.session_state.edit_recipe_id = recipe['id']
                    st.session_state.page = 'edit_recipe'
                    st.rerun()
            
            with col3:
                if st.button("🗑️ Delete", key=f"delete_{recipe['id']}", use_container_width=True):
                    if st.checkbox(f"Confirm delete", key=f"confirm_{recipe['id']}"):
                        delete_recipe(recipe['id'])
                        st.success("✅ Recipe deleted!")
                        st.rerun()
            
            st.markdown("---")
            
def browse_recipes_page():
    st.title("📖 Browse Recipes")
    st.markdown("*Discover sustainable recipes from our community!*")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        min_eco_score = st.selectbox("🌍 Min Eco Score", [1, 5, 7, 8, 9], index=1)
    with col2:
        sort_by = st.selectbox("📊 Sort by", ["Latest", "Eco Score", "Title"])
    with col3:
        search_filter = st.text_input("🔍 Quick search", placeholder="Search recipes...")
    
    # Get recipes
    recipes_df = get_recipes(approved_only=False)
    
    if len(recipes_df) == 0:
        st.info("📝 No recipes available yet. Be the first to add one!")
        if st.button("➕ Add First Recipe"):
            st.session_state.page = 'add_recipe'
            st.rerun()
        return
    
    # Apply filters
    if search_filter:
        mask = (recipes_df['title'].str.contains(search_filter, case=False, na=False) |
                recipes_df['description'].str.contains(search_filter, case=False, na=False) |
                recipes_df['ingredients'].str.contains(search_filter, case=False, na=False))
        recipes_df = recipes_df[mask]
    
    recipes_df = recipes_df[recipes_df['eco_score'] >= min_eco_score]
    
    # Sort recipes
    if sort_by == "Eco Score":
        recipes_df = recipes_df.sort_values('eco_score', ascending=False)
    elif sort_by == "Title":
        recipes_df = recipes_df.sort_values('title')
    else:  # Latest
        recipes_df = recipes_df.sort_values('created_at', ascending=False)
    
    if len(recipes_df) == 0:
        st.warning("🔍 No recipes match your filters. Try adjusting them.")
        return
    
    st.info(f"📊 Found {len(recipes_df)} recipe(s)")
    
    # Display recipes in grid
    cols_per_row = 2
    for i in range(0, len(recipes_df), cols_per_row):
        cols = st.columns(cols_per_row)
        
        for j, (_, recipe) in enumerate(recipes_df.iloc[i:i+cols_per_row].iterrows()):
            with cols[j]:
                with st.container():
                    # Recipe image
                    if recipe['image_data']:
                        try:
                            image_data = base64.b64decode(recipe['image_data'])
                            image = Image.open(io.BytesIO(image_data))
                            st.image(image, use_container_width=True)
                        except:
                            st.image("https://via.placeholder.com/300x200?text=Recipe+Image", use_container_width=True)
                    else:
                        st.image("https://via.placeholder.com/300x200?text=No+Image", use_container_width=True)
                    
                    # Recipe details
                    st.subheader(recipe['title'])
                    st.caption(f"👤 By {recipe['username']} • 📅 {recipe['created_at'][:10]}")
                    
                    # Eco score with color
                    eco_score = recipe['eco_score']
                    if eco_score >= 8:
                        score_color = "🟢"
                    elif eco_score >= 6:
                        score_color = "🟡"
                    else:
                        score_color = "🟠"
                    
                    st.write(f"{score_color} Eco Score: **{eco_score}/10**")
                    
                    # Description (truncated)
                    description = recipe['description']
                    if len(description) > 100:
                        description = description[:100] + "..."
                    st.write(description)
                    
                    # View button
                    if st.button(f"👀 View Recipe", key=f"view_{recipe['id']}", use_container_width=True):
                        st.session_state.selected_recipe_id = recipe['id']
                        st.session_state.page = 'view_recipe'
                        st.rerun()
                    
                    st.markdown("---")

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
    
    st.title("👥 Manage Users")
    
    conn = sqlite3.connect('ecofood.db')
    users_df = pd.read_sql_query("SELECT id, username, email, is_admin, created_at FROM users", conn)
    
    if len(users_df) == 0:
        st.info("👤 No users found in the database.")
        conn.close()
        return
    
    st.info(f"📊 Total Users: {len(users_df)}")
    
    # Display users with action buttons
    for _, user in users_df.iterrows():
        with st.container():
            col1, col2, col3, col4 = st.columns([3, 2, 2, 2])
            
            with col1:
                admin_badge = "🔧 ADMIN" if user['is_admin'] else "👤 USER"
                st.write(f"**{user['username']}** {admin_badge}")
                st.caption(f"📧 {user['email']}")
            
            with col2:
                st.write(f"📅 Joined: {user['created_at'][:10]}")
                
                # Get user's recipe count
                recipe_count = pd.read_sql_query(
                    "SELECT COUNT(*) as count FROM recipes WHERE user_id = ?", 
                    conn, params=[user['id']]
                ).iloc[0]['count']
                st.caption(f"🍽️ {recipe_count} recipes")
            
            with col3:
                # Toggle admin status (but not for current admin)
                if user['id'] != st.session_state.user['id']:
                    if user['is_admin']:
                        if st.button(f"⬇️ Remove Admin", key=f"remove_admin_{user['id']}", help="Remove admin privileges"):
                            c = conn.cursor()
                            c.execute("UPDATE users SET is_admin = 0 WHERE id = ?", (user['id'],))
                            conn.commit()
                            log_activity(st.session_state.user['id'], "Admin Action", f"Removed admin privileges from {user['username']}")
                            st.success(f"✅ Removed admin privileges from {user['username']}")
                            st.rerun()
                    else:
                        if st.button(f"⬆️ Make Admin", key=f"make_admin_{user['id']}", help="Grant admin privileges"):
                            c = conn.cursor()
                            c.execute("UPDATE users SET is_admin = 1 WHERE id = ?", (user['id'],))
                            conn.commit()
                            log_activity(st.session_state.user['id'], "Admin Action", f"Granted admin privileges to {user['username']}")
                            st.success(f"✅ Granted admin privileges to {user['username']}")
                            st.rerun()
                else:
                    st.info("👑 Current Admin")
            
            with col4:
                # Delete user (but not current admin)
                if user['id'] != st.session_state.user['id']:
                    # Create unique key for confirmation checkbox
                    confirm_key = f"confirm_delete_{user['id']}"
                    
                    # Show delete button
                    if st.button(f"🗑️ Delete", key=f"delete_user_{user['id']}", help="Delete this user", type="secondary"):
                        st.session_state[f"show_confirm_{user['id']}"] = True
                    
                    # Show confirmation if delete was clicked
                    if st.session_state.get(f"show_confirm_{user['id']}", False):
                        st.warning(f"⚠️ Delete {user['username']}?")
                        
                        col4_1, col4_2 = st.columns(2)
                        with col4_1:
                            if st.button(f"✅ Yes", key=f"confirm_yes_{user['id']}", type="primary"):
                                try:
                                    c = conn.cursor()
                                    # Delete user's data first (foreign key constraints)
                                    c.execute("DELETE FROM recipes WHERE user_id = ?", (user['id'],))
                                    c.execute("DELETE FROM food_detections WHERE user_id = ?", (user['id'],))
                                    c.execute("DELETE FROM donations WHERE user_id = ?", (user['id'],))
                                    c.execute("DELETE FROM activity_logs WHERE user_id = ?", (user['id'],))
                                    # Finally delete the user
                                    c.execute("DELETE FROM users WHERE id = ?", (user['id'],))
                                    conn.commit()
                                    
                                    # Log the action
                                    log_activity(st.session_state.user['id'], "Admin Action", f"Deleted user {user['username']} and all associated data")
                                    
                                    st.success(f"✅ Deleted user {user['username']} and all their data")
                                    
                                    # Clear the confirmation state
                                    if f"show_confirm_{user['id']}" in st.session_state:
                                        del st.session_state[f"show_confirm_{user['id']}"]
                                    
                                    st.rerun()
                                    
                                except Exception as e:
                                    st.error(f"❌ Error deleting user: {str(e)}")
                        
                        with col4_2:
                            if st.button(f"❌ No", key=f"confirm_no_{user['id']}"):
                                # Clear the confirmation state
                                if f"show_confirm_{user['id']}" in st.session_state:
                                    del st.session_state[f"show_confirm_{user['id']}"]
                                st.rerun()
                else:
                    st.info("🛡️ Protected")
            
            st.markdown("---")
    
    conn.close()
    
    # Optional: Add bulk actions
    st.markdown("### 🔧 Bulk Actions")
    col_bulk1, col_bulk2 = st.columns(2)
    
    with col_bulk1:
        if st.button("📊 Export User Data", help="Download user list as CSV"):
            csv = users_df.to_csv(index=False)
            st.download_button(
                label="⬇️ Download CSV",
                data=csv,
                file_name=f"ecofood_users_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    with col_bulk2:
        # Count non-admin users for bulk delete warning
        non_admin_count = len(users_df[(users_df['is_admin'] == False) & (users_df['id'] != st.session_state.user['id'])])
        if non_admin_count > 0:
            if st.button(f"⚠️ Delete All Regular Users ({non_admin_count})", help="Delete all non-admin users"):
                if st.checkbox("⚠️ I understand this will delete all regular users and their data", key="bulk_delete_confirm"):
                    try:
                        conn = sqlite3.connect('ecofood.db')
                        c = conn.cursor()
                        
                        # Get list of regular users (excluding current admin)
                        regular_users = users_df[(users_df['is_admin'] == False) & (users_df['id'] != st.session_state.user['id'])]
                        
                        for _, reg_user in regular_users.iterrows():
                            # Delete all user data
                            c.execute("DELETE FROM recipes WHERE user_id = ?", (reg_user['id'],))
                            c.execute("DELETE FROM food_detections WHERE user_id = ?", (reg_user['id'],))
                            c.execute("DELETE FROM donations WHERE user_id = ?", (reg_user['id'],))
                            c.execute("DELETE FROM activity_logs WHERE user_id = ?", (reg_user['id'],))
                            c.execute("DELETE FROM users WHERE id = ?", (reg_user['id'],))
                        
                        conn.commit()
                        conn.close()
                        
                        log_activity(st.session_state.user['id'], "Admin Action", f"Bulk deleted {non_admin_count} regular users")
                        st.success(f"✅ Deleted {non_admin_count} regular users and all their data")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Error in bulk delete: {str(e)}")

def approve_recipes_page():
    if not st.session_state.user['is_admin']:
        st.error("Access denied. Admin privileges required.")
        return
    
    st.title("✅ Approve Recipes")
    
    pending_recipes = get_recipes(approved_only=False)
    pending_recipes = pending_recipes[pending_recipes['approved'] == False]
    
    if len(pending_recipes) > 0:
        for _, recipe in pending_recipes.iterrows():
            with st.container():
                st.subheader(recipe['title'])
                st.write(f"By: {recipe['username']}")
                st.write(f"Description: {recipe['description']}")
                st.write(f"Eco Score: {recipe['eco_score']}/10")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(f"✅ Approve", key=f"approve_{recipe['id']}"):
                        conn = sqlite3.connect('ecofood.db')
                        c = conn.cursor()
                        c.execute("UPDATE recipes SET approved = ? WHERE id = ?", (True, recipe['id']))
                        conn.commit()
                        conn.close()
                        st.success("Recipe approved!")
                        st.rerun()
                
                with col2:
                    if st.button(f"❌ Reject", key=f"reject_{recipe['id']}"):
                        conn = sqlite3.connect('ecofood.db')
                        c = conn.cursor()
                        c.execute("DELETE FROM recipes WHERE id = ?", (recipe['id'],))
                        conn.commit()
                        conn.close()
                        st.success("Recipe rejected!")
                        st.rerun()
                
                st.markdown("---")
    else:
        st.info("No pending recipes to review.")

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

def analytics_page():
    if not st.session_state.user['is_admin']:
        st.error("Access denied. Admin privileges required.")
        return
    
    st.title("📈 Analytics Dashboard")
    
    # Food detection analytics
    st.subheader("📷 Food Detection Analytics")
    
    conn = sqlite3.connect('ecofood.db')
    
    # Detection stats
    col1, col2, col3 = st.columns(3)
    
    with col1:
        detection_count = pd.read_sql_query("SELECT COUNT(*) as count FROM food_detections", conn).iloc[0]['count']
        st.metric("Total Scans", detection_count)
    
    with col2:
        unique_users = pd.read_sql_query("SELECT COUNT(DISTINCT user_id) as count FROM food_detections WHERE user_id IS NOT NULL", conn).iloc[0]['count']
        st.metric("Active Scanners", unique_users)
    
    with col3:
        avg_items = pd.read_sql_query("SELECT AVG(JSON_ARRAY_LENGTH(detected_items)) as avg FROM food_detections", conn).iloc[0]['avg']
        if avg_items:
            st.metric("Avg Items/Scan", f"{avg_items:.1f}")
        else:
            st.metric("Avg Items/Scan", "0")
    
    # Recent detections
    st.subheader("🕒 Recent Food Detections")
    recent_detections = pd.read_sql_query("""
        SELECT fd.*, u.username 
        FROM food_detections fd 
        LEFT JOIN users u ON fd.user_id = u.id 
        ORDER BY fd.created_at DESC 
        LIMIT 10
    """, conn)
    
    if len(recent_detections) > 0:
        for _, detection in recent_detections.iterrows():
            detected_items = json.loads(detection['detected_items'])
            user_display = detection['username'] if detection['username'] else "Anonymous"
            st.write(f"👤 **{user_display}** detected: {', '.join(detected_items)} - {detection['created_at']}")
    else:
        st.info("No food detections yet!")
    
    conn.close()

def monitor_activity_page():
    if not st.session_state.user['is_admin']:
        st.error("Access denied. Admin privileges required.")
        return
    
    st.title("📊 Monitor Activity")
    
    conn = sqlite3.connect('ecofood.db')
    
    # Recent activity
    activity_df = pd.read_sql_query("""
        SELECT al.*, u.username 
        FROM activity_logs al 
        JOIN users u ON al.user_id = u.id 
        ORDER BY al.timestamp DESC 
        LIMIT 50
    """, conn)
    
    if len(activity_df) > 0:
        st.dataframe(activity_df[['username', 'action', 'details', 'timestamp']], use_container_width=True)
    else:
        st.info("No activity logs yet.")
    
    conn.close()

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
    elif st.session_state.page == 'food_detection':
        food_detection_page()
    elif st.session_state.page == 'add_recipe':
        if st.session_state.user:
            add_recipe_page()
        else:
            st.error("Please login to add recipes.")
    elif st.session_state.page == 'my_recipes':
        my_recipes_page()
    elif st.session_state.page == 'view_recipe':
        view_recipe_page()
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
    elif st.session_state.page == 'analytics':
        analytics_page()
    elif st.session_state.page == 'monitor_activity':
        monitor_activity_page()
    elif st.session_state.page == 'help':
        st.title("❓ Help & Support")
        st.markdown("""
        **Welcome to EcoFood Help Center!**
        
        **Getting Started:**
        - Register for a free account
        - Try our new AI food detection feature
        - Add your first sustainable recipe
        - Browse recipes from our community
        
        **Food Detection Feature:**
        - Take a photo of your ingredients
        - Our AI will identify food items
        - Get personalized recipe recommendations
        - Edit detected items for accuracy
        - Save recommended recipes to your collection
        
        **Features:**
        - Recipe sharing with eco-scoring
        - AI-powered food detection
        - Community donations
        - Recipe search and filtering
        - User profiles and activity tracking
        
        **Need More Help?**
        Contact our support team through the Contact Support page.
        """)
    elif st.session_state.page == 'contact_support':
        st.title("📞 Contact Support")
        st.markdown("*Get help with EcoFood platform*")
        
        # Direct contact information
        st.markdown("---")
        st.subheader("📬 Direct Contact Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **📧 Email Support:**
            - **Email:** [ecofood1401@gmail.com](mailto:ecofood1401@gmail.com)
            - **Response Time:** Within 24 hours
            - **Best for:** Technical issues, account problems, suggestions
            """)
            
            # Clickable email button
            if st.button("📧 Send Email", use_container_width=True):
                st.markdown("**Click here to open your email client:**")
                st.markdown("📮 [ecofood1401@gmail.com](mailto:ecofood1401@gmail.com?subject=EcoFood%20Support%20Request)")
        
        with col2:
            st.markdown("""
            **📱 Phone Support:**
            - **Phone:** [010-380-8565](tel:+60103808565)
            - **Available:** Monday - Friday, 9 AM - 6 PM (MYT)
            - **Best for:** Urgent issues, immediate assistance
            """)
            
            # Phone contact info
            if st.button("📞 Call Now", use_container_width=True):
                st.info("📞 **Call:** 010-380-8565")
                st.caption("Available Monday - Friday, 9 AM - 6 PM (Malaysia Time)")
        
        # System administrator info
        st.markdown("---")
        st.subheader("👨‍💻 System Administrator")
        
        st.info("""
        **System Admin & Developer:** Iswary  
        **Specializes in:** Platform development, food detection AI, user management  
        **For technical inquiries:** Please mention your issue type in the subject line
        """)
        
    else:
        # Default to home
        home_page()

# Installation and setup instructions
def show_setup_instructions():
    st.sidebar.markdown("---")
    st.sidebar.markdown("**🔧 Setup Instructions**")
    
    with st.sidebar.expander("📦 Install Dependencies"):
        st.code("""
pip install streamlit pandas pillow
pip install ultralytics  # For YOLOv8
pip install openai       # For recipe recommendations
pip install opencv-python  # For image processing
        """)
    
    with st.sidebar.expander("🔑 API Keys Setup"):
        st.markdown("""
        **OpenAI API Key (Optional):**
        - Set environment variable: `OPENAI_API_KEY=your_key`
        - Or the app will use mock recommendations
        
        **YOLOv8 Model:**
        - App will download YOLOv8 automatically
        - For custom food model, place `food_detection_yolo.pt` in app directory
        """)

if __name__ == "__main__":
    # Show setup instructions in sidebar
    # show_setup_instructions()
    
    # Run main app
    main()