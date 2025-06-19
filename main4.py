from fastapi import FastAPI, HTTPException
from pathlib import Path
from fastapi.responses import FileResponse
from pydantic import BaseModel
import re
import json
import importlib
from diagrams import Diagram, Cluster, Edge
import logging
import time
from datetime import datetime
import traceback
import os
from fastapi.middleware.cors import CORSMiddleware
import subprocess
import requests
import httpx
import signal
import sys
import atexit

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

# Create specific loggers
app_logger = logging.getLogger('fastapi_app')
diagram_logger = logging.getLogger('diagram_builder')
ollama_logger = logging.getLogger('ollama_service')

# Ollama configuration
OLLAMA_DIR = os.path.abspath("./Ollama")
OLLAMA_EXECUTABLE = os.path.join(OLLAMA_DIR, 'ollama.exe')
OLLAMA_SERVER_URL = "http://localhost:11434"
# OLLAMA_MODEL = "llama3.2"
OLLAMA_MODEL = "deepseek-r1:7b"
#OLLAMA_MODEL = "gemma3:latest"


# Global variable to track Ollama process
ollama_process = None

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def is_ollama_running():
    """Check if Ollama is already running"""
    try:
        response = httpx.get(OLLAMA_SERVER_URL, timeout=5.0)
        return response.status_code == 200
    except (httpx.RequestError, httpx.TimeoutException):
        return False

def start_ollama():
    """Start Ollama if not already running"""
    global ollama_process
    
    if is_ollama_running():
        ollama_logger.info("Ollama is already running")
        return True
    
    try:
        ollama_logger.info(f"Starting Ollama from: {OLLAMA_DIR}")
        ollama_process = subprocess.Popen(
            [OLLAMA_EXECUTABLE, "serve"],
            cwd=OLLAMA_DIR,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
        )
        ollama_logger.info("Ollama started successfully")
        
        # Wait for Ollama to fully start
        time.sleep(3)
        
        # Verify it's running
        max_retries = 10
        for i in range(max_retries):
            if is_ollama_running():
                ollama_logger.info("Ollama is ready to accept requests")
                return True
            time.sleep(1)
            ollama_logger.info(f"Waiting for Ollama to start... ({i+1}/{max_retries})")
        
        ollama_logger.error("Ollama failed to start within timeout")
        return False
        
    except Exception as e:
        ollama_logger.error(f"Failed to start Ollama: {e}")
        return False

def download_model(model_name=OLLAMA_MODEL):
    """Download Ollama model if not already available"""
    endpoint = f"{OLLAMA_SERVER_URL}/api/pull"
    payload = {"model": model_name}

    try:
        ollama_logger.info(f"Checking if model '{model_name}' is available...")
        
        # First check if model already exists
        list_response = requests.get(f"{OLLAMA_SERVER_URL}/api/tags", timeout=10)
        if list_response.status_code == 200:
            models = list_response.json().get("models", [])
            for model in models:
                if model.get("name", "").startswith(model_name):
                    ollama_logger.info(f"Model '{model_name}' already exists")
                    return True

        ollama_logger.info(f"Downloading model '{model_name}'...")
        response = requests.post(endpoint, json=payload, stream=True, timeout=300)
        response.raise_for_status()

        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line.decode("utf-8"))
                    status = data.get("status", "")
                    
                    if "pulling" in status and "completed" in data and "total" in data:
                        completed = data["completed"]
                        total = data["total"]
                        percentage = (completed / total) * 100
                        ollama_logger.info(f"Download progress: {percentage:.1f}% ({completed}/{total} bytes)")
                    elif status == "success":
                        ollama_logger.info("Model download completed successfully!")
                        return True
                    else:
                        ollama_logger.debug(f"Status: {status}")
                        
                except json.JSONDecodeError as e:
                    ollama_logger.warning(f"JSON decode error: {e}")
                    
        return True
                    
    except requests.exceptions.Timeout:
        ollama_logger.error("Model download timeout occurred")
        return False
    except requests.exceptions.HTTPError as err:
        ollama_logger.error(f"HTTP error during model download: {err}")
        return False
    except Exception as err:
        ollama_logger.error(f"Error during model download: {err}")
        return False

def make_inference(model_name, prompt):
    """Make inference request to Ollama"""
    endpoint = f"{OLLAMA_SERVER_URL}/api/generate"
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "top_p": 0.9,
            "num_ctx": 8192,
            "num_predict": 4096,
        }
    }
    
    try:
        ollama_logger.info("Making inference request...")
        start_time = time.time()
        
        response = requests.post(endpoint, json=payload, timeout=360)
        response.raise_for_status()
        
        end_time = time.time()
        inference_time = end_time - start_time
        
        data = response.json()
        
        ollama_logger.info(f"Inference completed in {inference_time:.2f} seconds")
        
        if "response" in data:
            return data["response"], inference_time
        else:
            ollama_logger.error("No response found in the data")
            return None, inference_time
            
    except requests.exceptions.Timeout:
        ollama_logger.error("Inference timeout occurred")
        return None, 0
    except requests.exceptions.HTTPError as err:
        ollama_logger.error(f"HTTP error during inference: {err}")
        return None, 0
    except Exception as err:
        ollama_logger.error(f"Error during inference: {err}")
        return None, 0

def shutdown_ollama():
    """Shutdown Ollama process"""
    global ollama_process
    if ollama_process:
        ollama_logger.info("Shutting down Ollama process...")
        ollama_process.terminate()
        ollama_process = None

def signal_handler(sig, frame):
    """Handle graceful shutdown"""
    app_logger.info("Shutting down gracefully...")
    shutdown_ollama()
    sys.exit(0)

# Register signal handlers and cleanup
signal.signal(signal.SIGINT, signal_handler)
atexit.register(shutdown_ollama)

def import_icon(icon_str):
    """Import icon class from diagrams module"""
    try:
        parts = icon_str.split('.')
        module_path = '.'.join(parts[:-1])
        class_name = parts[-1]
        module = importlib.import_module(f"diagrams.{module_path}")
        return getattr(module, class_name)
    except Exception as e:
        diagram_logger.error(f"Failed to import icon {icon_str}: {str(e)}")
        raise

def build_diagram(data):
    """Build diagram from JSON data"""
    start_time = time.time()
    diagram_logger.info(f"Starting diagram build process")
    
    try:
        nodes_map = {}
        diagram_title = data.get("title", "Architecture Diagram")
        filename = diagram_title.lower().replace(" ", "_")

        diagram_logger.info(f"Building diagram: {diagram_title}")
        
        with Diagram(diagram_title, show=False, direction="TB", graph_attr={
            "fontsize": "20",       
            "fontname": "Helvetica",  
            "labelloc": "t",         
        }):
            group_clusters = {}
            
            # Create clusters
            for group in data.get("groups", []):
                group_clusters[group["name"]] = Cluster(
                    (group["name"]), 
                    graph_attr={
                        "margin": "50", 
                        "pad": "100", 
                        "fontsize": "14",
                        "style": "rounded,filled",
                    }
                )
                diagram_logger.debug(f"Created cluster: {group['name']}")

            # Create nodes inside clusters or directly in diagram
            for node in data.get("nodes", []):
                try:
                    icon_class = import_icon(node["icon"])
                    group_name = node.get("group")

                    if group_name and group_name in group_clusters:
                        with group_clusters[group_name]:
                            nodes_map[node["id"]] = icon_class(node["label"])
                    else:
                        nodes_map[node["id"]] = icon_class(node["label"])
                    
                    diagram_logger.debug(f"Created node: {node['id']} with icon {node['icon']}")
                except Exception as e:
                    diagram_logger.error(f"Failed to create node {node['id']}: {str(e)}")
                    raise

            # Create edges with styles
            for edge in data.get("edges", []):
                try:
                    from_node = nodes_map.get(edge["from"])
                    to_node = nodes_map.get(edge["to"])
                    
                    if from_node and to_node:
                        edge_args = {}
                        if "color" in edge:
                            edge_args["color"] = edge["color"]
                        if "style" in edge:
                            edge_args["style"] = edge["style"]
                        if "label" in edge:
                            edge_args["label"] = edge["label"]
                        
                        from_node >> Edge(**edge_args) >> to_node
                        diagram_logger.debug(f"Created edge: {edge['from']} -> {edge['to']}")
                    else:
                        diagram_logger.warning(f"Skipping edge due to missing nodes: {edge['from']} -> {edge['to']}")
                except Exception as e:
                    diagram_logger.error(f"Failed to create edge {edge['from']} -> {edge['to']}: {str(e)}")
                    continue
        
        output_file = f"{filename}.png"
        build_time = time.time() - start_time
        
        diagram_logger.info(f"Diagram built successfully: {output_file} (took {build_time:.2f}s)")
        return output_file
        
    except Exception as e:
        build_time = time.time() - start_time
        diagram_logger.error(f"Diagram build failed after {build_time:.2f}s: {str(e)}")
        diagram_logger.error(f"Traceback: {traceback.format_exc()}")
        raise

class PromptInput(BaseModel):
    prompt: str

@app.on_event("startup")
async def startup_event():
    """Initialize Ollama on startup"""
    app_logger.info("Starting up FastAPI application...")
    
    # Start Ollama
    if not start_ollama():
        app_logger.error("Failed to start Ollama - application may not work properly")
        return
    
    # Download model
    if not download_model(OLLAMA_MODEL):
        app_logger.error(f"Failed to download model {OLLAMA_MODEL} - application may not work properly")
        return
    
    app_logger.info("FastAPI application startup completed successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    app_logger.info("Shutting down FastAPI application...")
    shutdown_ollama()
    app_logger.info("FastAPI application shutdown completed")

@app.post("/generateDataFlowDiagram")
def process_prompt(input_data: PromptInput):
    """Process prompt and generate data flow diagram"""
    request_start_time = time.time()
    request_timestamp = datetime.now()
    
    app_logger.info(f"New request received at {request_timestamp}")
    app_logger.info(f"Input prompt length: {len(input_data.prompt)} characters")
    app_logger.debug(f"Input prompt: {input_data.prompt[:200]}...")
    
    try:
        # Step 1: Ensure Ollama is running and model is available
        if not is_ollama_running():
            app_logger.warning("Ollama not running, attempting to start...")
            if not start_ollama():
                raise HTTPException(status_code=500, detail="Failed to start Ollama")
            
            if not download_model(OLLAMA_MODEL):
                raise HTTPException(status_code=500, detail="Failed to download Ollama model")
        
        # Step 2: Generate response
        ollama_logger.info("Generating response from Ollama")
        full_prompt =      full_prompt = """You are a System Architecture Assistant. Your primary goal is to analyze, understand, and generate structured JSON representations of web-based application architectures using a defined schema.

Each JSON architecture contains:

Fields

    title: A short name for the architecture.
    nodes: List of components (systems, services, databases).

    id: Unique identifier.
    icon: Component type (must use from predefined list).
    label: Human-readable name.
    group: Logical category (Frontend, Backend, Database, Monitoring, External, etc.)
    edges: Directed connections between nodes.

    from, to: Component IDs.
    label (optional): Describes data flow (e.g., API Call, Auth, File Upload). Strictly Don't repeat the same label for different edges.
    groups: Logical group definitions used to cluster nodes.

Your Tasks

Parse and understand nodes, edges, and groups.
Learn how this format visually and logically maps application architecture.
Use it to answer questions, generate new diagrams, or refine existing ones.
Strictly use icons from the list provided below.

MANDATORY REQUIREMENTS:
- ALWAYS include a "user" node with icon "onprem.client.Users" and group "frontend"
- ALWAYS include a client device node with group "frontend". Choose the appropriate icon based on user requirements:
  * Use "onprem.client.Client" for desktop/laptop applications or when no specific device is mentioned
  * Use "generic.device.Mobile" for mobile applications or when mobile is specified
  * Use "generic.device.Tablet" for tablet applications or when tablet is specified  
  
- ALWAYS include a frontend application node (using frontend framework icons like "programming.framework.React", "programming.framework.Angular", etc.) with group "frontend"
- ALWAYS create edges: user → client → frontend application
- This user-client-frontend flow must be present in every architecture, regardless of the specific requirements

Available Icons (icon values)

Browser
onprem.client.Users

Devices
onprem.client.Client
generic.device.Mobile
generic.device.Tablet


Programming Languages

programming.language.Python 
programming.language.Java 
programming.language.Nodejs
programming.framework.Flask
programming.framework.Fastapi

Frontend Frameworks

programming.framework.React 
programming.framework.Angular 
programming.framework.Vue 
programming.framework.Nextjs
programming.framework.Flutter
programming.framework.Graphql


firebase.base.Firebase

On-Premise Databases & Tools

onprem.database.Oracle 
onprem.database.Postgresql 
onprem.database.Mongodb 
onprem.database.Mysql 
onprem.inmemory.Redis 
onprem.monitoring.Dynatrace

onprem.monitoring.Prometheus 
onprem.monitoring.Grafana 

SaaS Identity Providers

saas.identity.Auth0 

AWS Components

aws.compute.EC2 
aws.storage.SimpleStorageServiceS3 
aws.database.RDS 
aws.network.ElasticLoadBalancing 
aws.security.IdentityAndAccessManagementIam 
aws.management.Cloudwatch 
aws.network.VPC 
aws.network.Route53 

Azure Components

azure.compute.AppServices 
azure.database.SQLDatabases 
azure.database.BlobStorage 
azure.identity.ActiveDirectory 
azure.monitor.Monitor 
azure.monitor.Metrics 
azure.network.LoadBalancers 
azure.network.DNSZones 
azure.network.VirtualNetworks 

GCP Components

gcp.compute.ComputeEngine 
gcp.database.SQL 
gcp.storage.Storage 
gcp.operations.Monitoring 
gcp.operations.Logging 
gcp.network.DNS 
gcp.network.LoadBalancing 
gcp.security.Iam 
gcp.network.VirtualPrivateCloud 

Android and iOS 
programming.language.Kotlin
programming.language.Swift
programming.framework.Flutter   
programming.language.Ruby

Architecture JSON Example

Title: E-Commerce Platform with Microservices and Monitoring

json
{
"title": "E-Commerce Web Application with Microservices",
"nodes": [
    { "id": "user", "icon": "onprem.client.Users", "label": "User", "group": "frontend" },
    { "id": "device", "icon": "onprem.client.Client", "label": "Laptop", "group": "frontend" },

    { "id": "reactApp", "icon": "programming.framework.React", "label": "React Frontend", "group": "frontend" },
    { "id": "loadBalancer", "icon": "aws.network.ElasticLoadBalancing", "label": "AWS Load Balancer", "group": "backend" },

    { "id": "productService", "icon": "programming.language.Nodejs", "label": "Product Service", "group": "backend" },
    { "id": "wishlistService", "icon": "programming.language.Nodejs", "label": "Wishlist Service", "group": "backend" },
    { "id": "cartService", "icon": "programming.language.Nodejs", "label": "Cart Service", "group": "backend" },
    { "id": "paymentService", "icon": "programming.language.Nodejs", "label": "Payment Service", "group": "backend" },

    { "id": "productDb", "icon": "onprem.database.Postgresql", "label": "Product DB", "group": "database" },
    { "id": "wishlistDb", "icon": "onprem.database.Mongodb", "label": "Wishlist DB", "group": "database" },
    { "id": "cartDb", "icon": "onprem.database.Mysql", "label": "Cart DB", "group": "database" },
    { "id": "paymentDb", "icon": "onprem.database.Oracle", "label": "Payment DB", "group": "database" },

    { "id": "prometheus", "icon": "onprem.monitoring.Prometheus", "label": "Prometheus", "group": "monitoring" },
    { "id": "grafana", "icon": "onprem.monitoring.Grafana", "label": "Grafana", "group": "monitoring" }
],
"edges": [
    { "from": "user", "to": "device" },
     { "from": "device", "to": "reactApp" },
    { "from": "reactApp", "to": "loadBalancer" },

    { "from": "loadBalancer", "to": "productService", "label": "API Call" },
    { "from": "loadBalancer", "to": "wishlistService", "label": "API Call" },
    { "from": "loadBalancer", "to": "cartService", "label": "API Call" },
    { "from": "loadBalancer", "to": "paymentService", "label": "API Call" },

    { "from": "productService", "to": "productDb", "label": "Read/Write" },
    { "from": "wishlistService", "to": "wishlistDb", "label": "Read/Write" },
    { "from": "cartService", "to": "cartDb", "label": "Read/Write" },
    { "from": "paymentService", "to": "paymentDb", "label": "Read/Write" },

    { "from": "cartService", "to": "paymentService", "label": "Trigger Checkout" },

    { "from": "productService", "to": "prometheus", "label": "Metrics" },
    { "from": "wishlistService", "to": "prometheus", "label": "Metrics" },
    { "from": "cartService", "to": "prometheus", "label": "Metrics" },
    { "from": "paymentService", "to": "prometheus", "label": "Metrics" },

    { "from": "prometheus", "to": "grafana", "label": "Visualize Metrics" }
],
"groups": [
    { "name": "frontend" },
    { "name": "backend" },
    { "name": "database" },
    { "name": "monitoring" }
]
}


Architecture JSON Example 2 : Web Application using Azure Microservices and Monitoring

    "title": "Healthcare Appointment Booking System",
  "nodes": [
    { "id": "user", "icon": "onprem.client.Users", "label": "Patient", "group": "frontend" },
    { "id": "mobile", "icon": "generic.device.Mobile", "label": "Mobile App", "group": "frontend" },
    { "id": "angularApp", "icon": "programming.framework.Angular", "label": "Angular Web App", "group": "frontend" },

    { "id": "azureLoadBalancer", "icon": "azure.network.LoadBalancers", "label": "Azure Load Balancer", "group": "backend" },

    { "id": "bookingService", "icon": "programming.language.Java", "label": "Booking Service", "group": "backend" },
    { "id": "doctorService", "icon": "programming.language.Python", "label": "Doctor Service", "group": "backend" },
    { "id": "notificationService", "icon": "programming.language.Nodejs", "label": "Notification Service", "group": "backend" },

    { "id": "appointmentDb", "icon": "azure.database.SQLDatabases", "label": "Appointments DB", "group": "database" },
    { "id": "doctorDb", "icon": "onprem.database.Postgresql", "label": "Doctor DB", "group": "database" },

    { "id": "monitoring", "icon": "azure.monitor.Monitor", "label": "Azure Monitor", "group": "monitoring" },
    { "id": "metrics", "icon": "azure.monitor.Metrics", "label": "Azure Metrics", "group": "monitoring" }
  ],
  "edges": [
    { "from": "user", "to": "mobile" },
    { "from": "user", "to": "angularApp" },

    { "from": "mobile", "to": "azureLoadBalancer" },
    { "from": "angularApp", "to": "azureLoadBalancer" },

    { "from": "azureLoadBalancer", "to": "bookingService", "label": "Route - Booking" },
    { "from": "azureLoadBalancer", "to": "doctorService", "label": "Route - Doctors" },
    { "from": "azureLoadBalancer", "to": "notificationService", "label": "Route - Notifications" },

    { "from": "bookingService", "to": "appointmentDb", "label": "DB Ops - Booking" },
    { "from": "doctorService", "to": "doctorDb", "label": "DB Ops - Doctor" },

    { "from": "bookingService", "to": "monitoring", "label": "Push Metrics - Booking" },
    { "from": "doctorService", "to": "monitoring", "label": "Push Metrics - Doctor" },
    { "from": "notificationService", "to": "monitoring", "label": "Push Metrics - Notifications" },

    { "from": "monitoring", "to": "metrics", "label": "Show Metrics" }
  ],
  "groups": [
    { "name": "frontend" },
    { "name": "backend" },
    { "name": "database" },
    { "name": "monitoring" }
  ]

  
   Architecture JSON Example 3 - Mobile Content App with Azure Microservices & Firebase
    {
  "title": "Mobile Content App with Azure Microservices & Firebase",
  "nodes": [
    { "id": "user", "icon": "onprem.client.Users", "label": "User", "group": "frontend" },
    { "id": "mobileDevice", "icon": "generic.device.Mobile", "label": "Mobile Device", "group": "frontend" },
    { "id": "tabletDevice", "icon": "generic.device.Tablet", "label": "Tablet Device", "group": "frontend" },
    { "id": "kotlinApp", "icon": "programming.language.Kotlin", "label": "Kotlin Mobile App", "group": "frontend" },

    { "id": "azureLoadBalancer", "icon": "azure.network.LoadBalancers", "label": "Azure Load Balancer", "group": "backend" },

    { "id": "authService", "icon": "azure.compute.AppServices", "label": "Auth Service", "group": "backend" },
    { "id": "contentService", "icon": "azure.compute.AppServices", "label": "Content Service", "group": "backend" },
    { "id": "notificationService", "icon": "azure.compute.AppServices", "label": "Notification Service", "group": "backend" },

    { "id": "aad", "icon": "azure.identity.ActiveDirectory", "label": "Azure AD", "group": "identity" },
    { "id": "contentDb", "icon": "azure.database.SQLDatabases", "label": "Content DB", "group": "database" },
    { "id": "mediaStorage", "icon": "azure.database.BlobStorage", "label": "Blob Storage", "group": "database" },
    { "id": "notificationDb", "icon": "azure.database.SQLDatabases", "label": "Notification DB", "group": "database" },

    { "id": "firebase", "icon": "onprem.monitoring.Dynatrace", "label": "Firebase Notifications", "group": "external" },

    { "id": "azureMonitor", "icon": "azure.monitor.Monitor", "label": "Azure Monitor", "group": "monitoring" },
    { "id": "azureMetrics", "icon": "azure.monitor.Metrics", "label": "Azure Metrics", "group": "monitoring" },
    { "id": "dns", "icon": "azure.network.DNSZones", "label": "Azure DNS", "group": "external" }
  ],
  "edges": [
    { "from": "user", "to": "mobileDevice" },
    { "from": "user", "to": "tabletDevice" },

    { "from": "mobileDevice", "to": "kotlinApp" },
    { "from": "tabletDevice", "to": "kotlinApp" },

    { "from": "kotlinApp", "to": "azureLoadBalancer", "label": "Send Requests" },

    { "from": "azureLoadBalancer", "to": "authService", "label": "Route to Auth" },
    { "from": "azureLoadBalancer", "to": "contentService", "label": "Route to Content" },
    { "from": "azureLoadBalancer", "to": "notificationService", "label": "Route to Notify" },

    { "from": "authService", "to": "aad", "label": "User Authentication" },
    { "from": "contentService", "to": "contentDb", "label": "Access DB - Content" },
    { "from": "contentService", "to": "mediaStorage", "label": "Access Storage - Media" },
    { "from": "notificationService", "to": "notificationDb", "label": "Access DB - Notify" },
    { "from": "notificationService", "to": "firebase", "label": "Push Notifications" },

    { "from": "authService", "to": "azureMonitor", "label": "Monitor - Auth" },
    { "from": "contentService", "to": "azureMonitor", "label": "Monitor - Content" },
    { "from": "notificationService", "to": "azureMonitor", "label": "Monitor - Notify" },

    { "from": "azureMonitor", "to": "azureMetrics", "label": "Emit Metrics" },

    { "from": "kotlinApp", "to": "dns", "label": "Resolve Domain" }
  ],
  "groups": [
    { "name": "frontend" },
    { "name": "backend" },
    { "name": "database" },
    { "name": "monitoring" },
    { "name": "identity" },
    { "name": "external" }
  ]
}



REMEMBER: Every architecture JSON you generate must include:
1. A user node: { "id": "user", "icon": "onprem.client.Users", "label": "User", "group": "frontend" }
2. A client device node with appropriate icon and group "frontend"
3. A frontend application node with appropriate framework icon and group "frontend"
4. Edges connecting: user → client → frontend application
5. Make sure to use the correct icons from the provided list



""" + input_data.prompt
        # Create a JSON for a web application with the following architecture:

        ollama_logger.debug(f"Full prompt length: {len(full_prompt)} characters")
        app_logger.info(f"Full prompt Input: {len(full_prompt)} chars")

        
        response_text, inference_time = make_inference(OLLAMA_MODEL, full_prompt)
        app_logger.debug(f"Response content: {response_text}...")

        
        if not response_text:
            raise HTTPException(status_code=500, detail="Failed to generate response from Ollama")
        
        ollama_logger.info(f"Response generated successfully in {inference_time:.2f}s")
        ollama_logger.debug(f"Response length: {len(response_text)} characters")
        app_logger.debug(f"Response content: {response_text}...")

        # Step 3: Process response and extract JSON
        app_logger.info("Processing Ollama response and extracting JSON")

         
        valid_icons = {
            "Browser", "onprem.client.Users", "onprem.client.Client", 
            "programming.language.Python", "programming.language.Java", "programming.language.Nodejs",
            "programming.framework.Flask", "programming.framework.Fastapi", "programming.framework.React",
            "programming.framework.Angular", "programming.framework.Vue", "programming.language.Kotlin",
            "programming.language.Swift", "programming.framework.Flutter", "programming.language.Ruby",
            "generic.device.Mobile", "generic.device.Tablet", "programming.framework.Nextjs","programming.framework.Flutter","programming.framework.Graphql","firebase.base.Firebase"
            "onprem.database.Oracle", "onprem.database.Postgresql", "onprem.database.Mongodb",
            "onprem.database.Mysql", "onprem.inmemory.Redis", "onprem.monitoring.Dynatrace",
            "onprem.monitoring.Prometheus", "onprem.monitoring.Grafana", "saas.identity.Auth0",
            "aws.compute.EC2", "aws.storage.SimpleStorageServiceS3", "aws.database.RDS",
            "aws.network.ElasticLoadBalancing", "aws.security.IdentityAndAccessManagementIam",
            "aws.management.Cloudwatch", "aws.network.VPC", "aws.network.Route53",
            "azure.compute.AppServices", "azure.database.SQLDatabases", "azure.database.BlobStorage",
            "azure.identity.ActiveDirectory", "azure.monitor.Monitor", "azure.monitor.Metrics",
            "azure.network.LoadBalancers", "azure.network.DNSZones", "azure.network.VirtualNetworks",
            "gcp.compute.ComputeEngine", "gcp.database.SQL", "gcp.storage.Storage",
            "gcp.operations.Monitoring", "gcp.operations.Logging", "gcp.network.DNS",
            "gcp.network.LoadBalancing", "gcp.security.Iam", "gcp.network.VirtualPrivateCloud"
        }
        app_logger.debug("Valid icons loaded for processing")
        match = re.search(r"```(?:json)?\n(.*?)```", response_text, re.DOTALL)
        print("JSON:  :")
        print(match)
        app_logger.info(f" LLM output: {response_text}")
        if not match:
            app_logger.error("No JSON code block found in Ollama response")
            app_logger.debug(f"Response content: {response_text[:500]}...")
            raise HTTPException(status_code=500, detail="No JSON code block found in response")

        # Step 4: Parse and clean JSON
        json_processing_start = time.time()
        
        try:
            json_block = match.group(1).strip()
            app_logger.debug(f"Extracted JSON block length: {len(json_block)} characters")
            
            # Remove comments
            json_block = re.sub(r'//.*', '', json_block)
            
            # Parse JSON
            json_data = json.loads(json_block)
            app_logger.info("JSON parsed successfully")
            
            # Validate and fix icons
            default_icon = "onprem.compute.Server"
            icon_fixes = 0
            
            for node in json_data.get("nodes", []):
                icon = node.get("icon", "")
                original_icon = icon

                # Check original icon
                if icon in valid_icons:
                    continue

                # Normalize: lowercase prefix + Capitalized suffix
                if "." in icon:
                    *prefix_parts, last = icon.split(".")
                    prefix = ".".join(part.lower() for part in prefix_parts)
                    last = last.capitalize()
                    normalized_icon = f"{prefix}.{last}"
                else:
                    normalized_icon = icon.lower()

                # Check normalized icon
                if normalized_icon in valid_icons:
                    node["icon"] = normalized_icon
                    if original_icon != normalized_icon:
                        icon_fixes += 1
                        app_logger.debug(f"Fixed icon: {original_icon} -> {normalized_icon}")
                else:
                    node["icon"] = default_icon
                    icon_fixes += 1
                    app_logger.warning(f"Invalid icon replaced: {original_icon} -> {default_icon}")

            if icon_fixes > 0:
                app_logger.info(f"Fixed {icon_fixes} invalid icons")

            # Deduplicate edge labels
            seen_labels = set()
            label_deduplication = 0
            
            for edge in json_data.get("edges", []):
                label = edge.get("label", "")
                if label and label in seen_labels:
                    edge["label"] = ""
                    label_deduplication += 1
                elif label:
                    seen_labels.add(label)
            
            if label_deduplication > 0:
                app_logger.info(f"Deduplicated {label_deduplication} edge labels")

            json_processing_time = time.time() - json_processing_start
            app_logger.info(f"JSON processing completed in {json_processing_time:.2f}s")
            
            # Log the processed data structure
            diagram_title = json_data.get("title", "Architecture Diagram")
            node_count = len(json_data.get("nodes", []))
            edge_count = len(json_data.get("edges", []))
            group_count = len(json_data.get("groups", []))
            
            app_logger.info(f"Processed diagram: '{diagram_title}' with {node_count} nodes, {edge_count} edges, {group_count} groups")

        except json.JSONDecodeError as e:
            app_logger.error(f"JSON parsing failed: {str(e)}")
            app_logger.debug(f"Invalid JSON content: {json_block[:500]}...")
            raise HTTPException(status_code=500, detail="Invalid JSON format in response")
        except Exception as e:
            app_logger.error(f"JSON processing failed: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to process JSON response")

        # Step 5: Build diagram
        try:
            app_logger.info("Starting diagram generation")
            img_path = build_diagram(json_data)
            
            # Verify file exists
            if not os.path.exists(img_path):
                app_logger.error(f"Generated diagram file not found: {img_path}")
                raise HTTPException(status_code=500, detail="Diagram file generation failed")
            
            file_size = os.path.getsize(img_path)
            app_logger.info(f"Diagram file generated: {img_path} ({file_size} bytes)")
            
        except Exception as e:
            app_logger.error(f"Diagram generation failed: {str(e)}")
            app_logger.error(f"Traceback: {traceback.format_exc()}")
            raise HTTPException(status_code=500, detail="Failed to generate diagram")

        # Step 6: Successful completion
        total_time = time.time() - request_start_time
        completion_timestamp = datetime.now()
        
        app_logger.info(f"Request completed successfully at {completion_timestamp}")
        app_logger.info(f"Total processing time: {total_time:.2f}s")
        app_logger.info(f"Output file: {img_path}")
        
        # Log success summary
        app_logger.info("=" * 80)
        app_logger.info("REQUEST SUMMARY - SUCCESS")
        app_logger.info(f"Start Time: {request_timestamp}")
        app_logger.info(f"End Time: {completion_timestamp}")
        app_logger.info(f"Duration: {total_time:.2f}s")
        app_logger.info(f"Input: {len(input_data.prompt)} chars")
        app_logger.info(f"Output: {img_path} ({file_size} bytes)")
        app_logger.info(f"Diagram: {diagram_title} ({node_count}N, {edge_count}E, {group_count}G)")
        app_logger.info("=" * 80)
        
        return FileResponse(str(img_path))

    except HTTPException as he:
        # Re-raise HTTP exceptions
        total_time = time.time() - request_start_time
        completion_timestamp = datetime.now()
        
        app_logger.error(f"HTTP Exception at {completion_timestamp}: {he.detail}")
        app_logger.error(f"Request failed after {total_time:.2f}s")
        
        # Log error summary
        app_logger.error("=" * 80)
        app_logger.error("REQUEST SUMMARY - HTTP ERROR")
        app_logger.error(f"Start Time: {request_timestamp}")
        app_logger.error(f"End Time: {completion_timestamp}")
        app_logger.error(f"Duration: {total_time:.2f}s")
        app_logger.error(f"Error: {he.detail}")
        app_logger.error("=" * 80)
        
        raise he
        
    except Exception as e:
        # Handle unexpected errors
        total_time = time.time() - request_start_time
        completion_timestamp = datetime.now()
        
        app_logger.error(f"Unexpected error at {completion_timestamp}: {str(e)}")
        app_logger.error(f"Request failed after {total_time:.2f}s")
        app_logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Log error summary
        app_logger.error("=" * 80)
        app_logger.error("REQUEST SUMMARY - UNEXPECTED ERROR")
        app_logger.error(f"Start Time: {request_timestamp}")
        app_logger.error(f"End Time: {completion_timestamp}")
        app_logger.error(f"Duration: {total_time:.2f}s")
        app_logger.error(f"Error: {str(e)}")
        app_logger.error("=" * 80)
        
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Health check endpoint with logging
@app.get("/health")
def health_check():
    """Health check endpoint"""
    timestamp = datetime.now()
    app_logger.info(f"Health check requested at {timestamp}")
    
    # Check if Ollama is running
    ollama_status = "running" if is_ollama_running() else "not running"
    
    return {
        "status": "healthy", 
        "timestamp": timestamp,
        "ollama_status": ollama_status
    }

if __name__ == "__main__":
    import uvicorn
    app_logger.info("Starting FastAPI application")
    uvicorn.run(app, host="0.0.0.0", port=8000)