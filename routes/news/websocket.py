from fastapi import WebSocket, WebSocketDisconnect, APIRouter
from typing import List, Dict, Any
import json
import logging
import asyncio
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/news/ws", tags=["news-websocket"])

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.subscriptions: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        # Remove from all subscriptions
        for topic, connections in self.subscriptions.items():
            if websocket in connections:
                connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        try:
            await websocket.send_text(message)
        except Exception as e:
            logger.error(f"Error sending personal message: {str(e)}")
            self.disconnect(websocket)

    async def broadcast(self, message: str):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"Error broadcasting message: {str(e)}")
                disconnected.append(connection)
        
        # Remove disconnected connections
        for connection in disconnected:
            self.disconnect(connection)

    async def subscribe(self, websocket: WebSocket, topic: str):
        if topic not in self.subscriptions:
            self.subscriptions[topic] = []
        self.subscriptions[topic].append(websocket)
        logger.info(f"Subscribed to topic: {topic}")

    async def unsubscribe(self, websocket: WebSocket, topic: str):
        if topic in self.subscriptions and websocket in self.subscriptions[topic]:
            self.subscriptions[topic].remove(websocket)
        logger.info(f"Unsubscribed from topic: {topic}")

    async def publish(self, topic: str, message: str):
        if topic in self.subscriptions:
            disconnected = []
            for connection in self.subscriptions[topic]:
                try:
                    await connection.send_text(message)
                except Exception as e:
                    logger.error(f"Error publishing to topic {topic}: {str(e)}")
                    disconnected.append(connection)
            
            # Remove disconnected connections
            for connection in disconnected:
                self.disconnect(connection)

manager = ConnectionManager()

@router.websocket("/")
async def websocket_endpoint(websocket: WebSocket):
    """Main WebSocket endpoint for real-time communication"""
    await manager.connect(websocket)
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Handle different message types
            message_type = message.get("type", "unknown")
            
            if message_type == "subscribe":
                topic = message.get("topic", "general")
                await manager.subscribe(websocket, topic)
                await manager.send_personal_message(
                    json.dumps({
                        "type": "subscription_confirmed",
                        "topic": topic,
                        "timestamp": datetime.now().isoformat()
                    }),
                    websocket
                )
            
            elif message_type == "unsubscribe":
                topic = message.get("topic", "general")
                await manager.unsubscribe(websocket, topic)
                await manager.send_personal_message(
                    json.dumps({
                        "type": "unsubscription_confirmed",
                        "topic": topic,
                        "timestamp": datetime.now().isoformat()
                    }),
                    websocket
                )
            
            elif message_type == "prediction_request":
                # Handle real-time prediction request
                await handle_prediction_request(websocket, message)
            
            elif message_type == "ping":
                await manager.send_personal_message(
                    json.dumps({
                        "type": "pong",
                        "timestamp": datetime.now().isoformat()
                    }),
                    websocket
                )
            
            else:
                await manager.send_personal_message(
                    json.dumps({
                        "type": "error",
                        "message": f"Unknown message type: {message_type}",
                        "timestamp": datetime.now().isoformat()
                    }),
                    websocket
                )
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        manager.disconnect(websocket)

async def handle_prediction_request(websocket: WebSocket, message: Dict[str, Any]):
    """Handle real-time prediction requests"""
    try:
        # Extract prediction data
        prediction_data = message.get("data", {})
        
        # Send acknowledgment
        await manager.send_personal_message(
            json.dumps({
                "type": "prediction_received",
                "request_id": message.get("request_id", str(uuid.uuid4())),
                "timestamp": datetime.now().isoformat()
            }),
            websocket
        )
        
        # Simulate prediction processing (in production, call actual prediction function)
        await asyncio.sleep(1)  # Simulate processing time
        
        # Send prediction result
        prediction_result = {
            "type": "prediction_result",
            "request_id": message.get("request_id", str(uuid.uuid4())),
            "data": {
                "spike_prediction": "SPIKE",
                "spike_confidence": 0.75,
                "hotel_predictions": {
                    "AHUN.N0000": {
                        "prediction": "up",
                        "confidence": 0.82,
                        "probabilities": {"up": 0.82, "down": 0.08, "neutral": 0.10}
                    },
                    "TRAN.N0000": {
                        "prediction": "neutral",
                        "confidence": 0.65,
                        "probabilities": {"up": 0.25, "down": 0.10, "neutral": 0.65}
                    }
                }
            },
            "timestamp": datetime.now().isoformat()
        }
        
        await manager.send_personal_message(json.dumps(prediction_result), websocket)
        
    except Exception as e:
        logger.error(f"Error handling prediction request: {str(e)}")
        await manager.send_personal_message(
            json.dumps({
                "type": "error",
                "message": f"Error processing prediction: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }),
            websocket
        )

# Background task for sending periodic updates
async def send_periodic_updates():
    """Send periodic system updates to all connected clients"""
    while True:
        try:
            # System status update
            status_update = {
                "type": "system_status",
                "data": {
                    "active_connections": len(manager.active_connections),
                    "total_subscriptions": sum(len(conns) for conns in manager.subscriptions.values()),
                    "system_health": "healthy",
                    "models_loaded": True
                },
                "timestamp": datetime.now().isoformat()
            }
            
            await manager.broadcast(json.dumps(status_update))
            
            # Wait for 30 seconds before next update
            await asyncio.sleep(30)
            
        except Exception as e:
            logger.error(f"Error sending periodic updates: {str(e)}")
            await asyncio.sleep(30)

# Start background task when the module loads
@router.on_event("startup")
async def start_background_tasks():
    """Start background tasks for WebSocket functionality"""
    asyncio.create_task(send_periodic_updates())

# WebSocket endpoints for specific topics
@router.websocket("/predictions")
async def predictions_websocket(websocket: WebSocket):
    """WebSocket endpoint specifically for prediction updates"""
    await manager.connect(websocket)
    await manager.subscribe(websocket, "predictions")
    
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@router.websocket("/alerts")
async def alerts_websocket(websocket: WebSocket):
    """WebSocket endpoint for system alerts and notifications"""
    await manager.connect(websocket)
    await manager.subscribe(websocket, "alerts")
    
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@router.websocket("/analytics")
async def analytics_websocket(websocket: WebSocket):
    """WebSocket endpoint for real-time analytics updates"""
    await manager.connect(websocket)
    await manager.subscribe(websocket, "analytics")
    
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# Helper functions for external use
async def broadcast_prediction(prediction_data: Dict[str, Any]):
    """Broadcast a prediction result to all subscribed clients"""
    message = {
        "type": "new_prediction",
        "data": prediction_data,
        "timestamp": datetime.now().isoformat()
    }
    await manager.publish("predictions", json.dumps(message))

async def broadcast_alert(alert_data: Dict[str, Any]):
    """Broadcast an alert to all subscribed clients"""
    message = {
        "type": "alert",
        "data": alert_data,
        "timestamp": datetime.now().isoformat()
    }
    await manager.publish("alerts", json.dumps(message))

async def broadcast_analytics(analytics_data: Dict[str, Any]):
    """Broadcast analytics update to all subscribed clients"""
    message = {
        "type": "analytics_update",
        "data": analytics_data,
        "timestamp": datetime.now().isoformat()
    }
    await manager.publish("analytics", json.dumps(message)) 