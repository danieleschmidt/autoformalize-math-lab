"""Distributed coordination system for scalable mathematical formalization.

This module provides distributed processing capabilities including:
- Load balancing and task distribution
- Horizontal scaling coordination
- Fault tolerance and failover
- Resource pooling and management
- Inter-node communication and synchronization
"""

import asyncio
import time
import json
import hashlib
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import uuid
import socket
from collections import defaultdict

from ..utils.logging_config import setup_logger


class NodeRole(Enum):
    """Roles that nodes can play in the distributed system."""
    COORDINATOR = "coordinator"
    WORKER = "worker"
    HYBRID = "hybrid"


class TaskStatus(Enum):
    """Status of distributed tasks."""
    PENDING = "pending"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class LoadBalancingStrategy(Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    CPU_BASED = "cpu_based"
    RESPONSE_TIME = "response_time"
    ADAPTIVE = "adaptive"


@dataclass
class NodeInfo:
    """Information about a node in the distributed system."""
    node_id: str
    role: NodeRole
    host: str
    port: int
    capabilities: Dict[str, Any]
    load_metrics: Dict[str, float] = field(default_factory=dict)
    last_heartbeat: float = 0.0
    status: str = "active"
    current_tasks: int = 0
    max_tasks: int = 10


@dataclass
class DistributedTask:
    """Represents a task in the distributed system."""
    task_id: str
    task_type: str
    payload: Dict[str, Any]
    priority: int = 1
    created_at: float = field(default_factory=time.time)
    assigned_node: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    retries: int = 0
    max_retries: int = 3
    timeout: float = 300.0  # 5 minutes


@dataclass
class ClusterConfig:
    """Configuration for distributed cluster."""
    cluster_name: str = "autoformalize-cluster"
    node_discovery_port: int = 8765
    heartbeat_interval: float = 10.0
    node_timeout: float = 30.0
    load_balancing_strategy: LoadBalancingStrategy = LoadBalancingStrategy.ADAPTIVE
    max_tasks_per_node: int = 10
    task_timeout: float = 300.0
    enable_failover: bool = True
    enable_auto_scaling: bool = True


class DistributedTaskManager:
    """Manages task distribution and execution across nodes."""
    
    def __init__(self, node_info: NodeInfo, config: ClusterConfig):
        """Initialize distributed task manager.
        
        Args:
            node_info: Information about this node
            config: Cluster configuration
        """
        self.node_info = node_info
        self.config = config
        self.logger = setup_logger(__name__)
        
        # Task management
        self.pending_tasks: Dict[str, DistributedTask] = {}
        self.running_tasks: Dict[str, DistributedTask] = {}
        self.completed_tasks: Dict[str, DistributedTask] = {}
        
        # Node management
        self.known_nodes: Dict[str, NodeInfo] = {}
        self.known_nodes[node_info.node_id] = node_info
        
        # Load balancing
        self.last_assigned_node = 0
        self.node_performance_history: Dict[str, List[float]] = defaultdict(list)
        
        # Communication
        self.server: Optional[asyncio.Server] = None
        self.client_connections: Dict[str, Tuple[asyncio.StreamReader, asyncio.StreamWriter]] = {}
        
        # Background tasks
        self.background_tasks: List[asyncio.Task] = []
        
        self.logger.info(f"Distributed task manager initialized for node {node_info.node_id}")
    
    async def start(self) -> None:
        """Start the distributed task manager."""
        try:
            # Start server for incoming connections
            self.server = await asyncio.start_server(
                self._handle_client,
                self.node_info.host,
                self.node_info.port
            )
            
            # Start background tasks
            self.background_tasks = [
                asyncio.create_task(self._heartbeat_loop()),
                asyncio.create_task(self._task_monitoring_loop()),
                asyncio.create_task(self._node_discovery_loop()),
                asyncio.create_task(self._load_balancing_loop())
            ]
            
            self.logger.info(f"Distributed task manager started on {self.node_info.host}:{self.node_info.port}")
            
        except Exception as e:
            self.logger.error(f"Failed to start distributed task manager: {e}")
            raise
    
    async def stop(self) -> None:
        """Stop the distributed task manager."""
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Close client connections
        for reader, writer in self.client_connections.values():
            writer.close()
            await writer.wait_closed()
        
        # Close server
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        
        self.logger.info("Distributed task manager stopped")
    
    async def submit_task(self, task_type: str, payload: Dict[str, Any], priority: int = 1) -> str:
        """Submit a task for distributed execution.
        
        Args:
            task_type: Type of task to execute
            payload: Task payload data
            priority: Task priority (higher = more important)
            
        Returns:
            Task ID
        """
        task_id = str(uuid.uuid4())
        task = DistributedTask(
            task_id=task_id,
            task_type=task_type,
            payload=payload,
            priority=priority
        )
        
        self.pending_tasks[task_id] = task
        
        # Try to assign immediately if we're a coordinator
        if self.node_info.role in [NodeRole.COORDINATOR, NodeRole.HYBRID]:
            await self._assign_task(task)
        
        self.logger.info(f"Task submitted: {task_id} (type: {task_type})")
        return task_id
    
    async def get_task_result(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """Get result of a distributed task.
        
        Args:
            task_id: ID of task to get result for
            timeout: Optional timeout in seconds
            
        Returns:
            Task result
            
        Raises:
            TimeoutError: If task doesn't complete within timeout
            RuntimeError: If task failed
        """
        start_time = time.time()
        timeout = timeout or self.config.task_timeout
        
        while True:
            # Check if task is completed
            if task_id in self.completed_tasks:
                task = self.completed_tasks[task_id]
                if task.status == TaskStatus.COMPLETED:
                    return task.result
                elif task.status == TaskStatus.FAILED:
                    raise RuntimeError(f"Task failed: {task.error}")
            
            # Check timeout
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Task {task_id} did not complete within {timeout} seconds")
            
            await asyncio.sleep(0.1)
    
    async def _assign_task(self, task: DistributedTask) -> bool:
        """Assign task to an appropriate node.
        
        Args:
            task: Task to assign
            
        Returns:
            True if task was assigned successfully
        """
        # Select best node for task
        target_node = await self._select_node_for_task(task)
        
        if not target_node:
            self.logger.warning(f"No suitable node found for task {task.task_id}")
            return False
        
        # Assign task
        task.assigned_node = target_node.node_id
        task.status = TaskStatus.ASSIGNED
        
        # Move to appropriate tracking
        if task.task_id in self.pending_tasks:
            del self.pending_tasks[task.task_id]
        self.running_tasks[task.task_id] = task
        
        # Send task to node (if not local)
        if target_node.node_id != self.node_info.node_id:
            success = await self._send_task_to_node(task, target_node)
            if not success:
                # Failed to send, put back in pending
                task.status = TaskStatus.PENDING
                task.assigned_node = None
                self.pending_tasks[task.task_id] = task
                del self.running_tasks[task.task_id]
                return False
        else:
            # Execute locally
            asyncio.create_task(self._execute_local_task(task))
        
        self.logger.info(f"Task {task.task_id} assigned to node {target_node.node_id}")
        return True
    
    async def _select_node_for_task(self, task: DistributedTask) -> Optional[NodeInfo]:
        """Select the best node for executing a task.
        
        Args:
            task: Task to assign
            
        Returns:
            Selected node or None if no suitable node found
        """
        available_nodes = [
            node for node in self.known_nodes.values()
            if (node.status == "active" and 
                node.current_tasks < node.max_tasks and
                self._node_can_handle_task(node, task))
        ]
        
        if not available_nodes:
            return None
        
        # Apply load balancing strategy
        if self.config.load_balancing_strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._round_robin_selection(available_nodes)
        elif self.config.load_balancing_strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return min(available_nodes, key=lambda n: n.current_tasks)
        elif self.config.load_balancing_strategy == LoadBalancingStrategy.CPU_BASED:
            return min(available_nodes, key=lambda n: n.load_metrics.get('cpu', 100))
        elif self.config.load_balancing_strategy == LoadBalancingStrategy.RESPONSE_TIME:
            return self._response_time_selection(available_nodes)
        else:  # ADAPTIVE
            return self._adaptive_selection(available_nodes, task)
    
    def _round_robin_selection(self, nodes: List[NodeInfo]) -> NodeInfo:
        """Round-robin node selection."""
        self.last_assigned_node = (self.last_assigned_node + 1) % len(nodes)
        return nodes[self.last_assigned_node]
    
    def _response_time_selection(self, nodes: List[NodeInfo]) -> NodeInfo:
        """Select node based on historical response times."""
        best_node = nodes[0]
        best_avg_time = float('inf')
        
        for node in nodes:
            if node.node_id in self.node_performance_history:
                history = self.node_performance_history[node.node_id]
                if history:
                    avg_time = sum(history[-10:]) / len(history[-10:])  # Last 10 measurements
                    if avg_time < best_avg_time:
                        best_avg_time = avg_time
                        best_node = node
        
        return best_node
    
    def _adaptive_selection(self, nodes: List[NodeInfo], task: DistributedTask) -> NodeInfo:
        """Adaptive node selection considering multiple factors."""
        scored_nodes = []
        
        for node in nodes:
            score = self._calculate_node_score(node, task)
            scored_nodes.append((score, node))
        
        # Select node with highest score
        scored_nodes.sort(reverse=True)
        return scored_nodes[0][1]
    
    def _calculate_node_score(self, node: NodeInfo, task: DistributedTask) -> float:
        """Calculate suitability score for a node."""
        score = 100.0  # Base score
        
        # Penalize high CPU usage
        cpu_usage = node.load_metrics.get('cpu', 0)
        score -= cpu_usage * 0.5
        
        # Penalize high current task load
        load_ratio = node.current_tasks / max(node.max_tasks, 1)
        score -= load_ratio * 30
        
        # Bonus for task-specific capabilities
        if task.task_type in node.capabilities:
            score += 20
        
        # Historical performance bonus
        if node.node_id in self.node_performance_history:
            history = self.node_performance_history[node.node_id]
            if history:
                avg_time = sum(history[-5:]) / len(history[-5:])  # Last 5 measurements
                if avg_time < 1.0:  # Fast response time
                    score += 15
                elif avg_time > 5.0:  # Slow response time
                    score -= 15
        
        return max(0, score)
    
    def _node_can_handle_task(self, node: NodeInfo, task: DistributedTask) -> bool:
        """Check if node can handle the given task type."""
        # Check basic capability
        if task.task_type not in node.capabilities:
            return False
        
        # Check resource requirements (if specified)
        task_requirements = task.payload.get('requirements', {})
        
        if 'min_memory' in task_requirements:
            if node.load_metrics.get('memory_available', 0) < task_requirements['min_memory']:
                return False
        
        if 'requires_gpu' in task_requirements:
            if not node.capabilities.get('gpu_available', False):
                return False
        
        return True
    
    async def _send_task_to_node(self, task: DistributedTask, node: NodeInfo) -> bool:
        """Send task to a remote node.
        
        Args:
            task: Task to send
            node: Target node
            
        Returns:
            True if task was sent successfully
        """
        try:
            # Get or create connection to node
            connection = await self._get_node_connection(node)
            if not connection:
                return False
            
            reader, writer = connection
            
            # Send task
            message = {
                'type': 'task_assignment',
                'task': {
                    'task_id': task.task_id,
                    'task_type': task.task_type,
                    'payload': task.payload,
                    'priority': task.priority,
                    'timeout': task.timeout
                }
            }
            
            data = json.dumps(message).encode() + b'\n'
            writer.write(data)
            await writer.drain()
            
            # Wait for acknowledgment
            response_data = await reader.readline()
            response = json.loads(response_data.decode().strip())
            
            if response.get('status') == 'accepted':
                node.current_tasks += 1
                return True
            else:
                self.logger.error(f"Node {node.node_id} rejected task: {response.get('error')}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to send task to node {node.node_id}: {e}")
            return False
    
    async def _get_node_connection(self, node: NodeInfo) -> Optional[Tuple[asyncio.StreamReader, asyncio.StreamWriter]]:
        """Get or create connection to a node."""
        if node.node_id in self.client_connections:
            return self.client_connections[node.node_id]
        
        try:
            reader, writer = await asyncio.open_connection(node.host, node.port)
            self.client_connections[node.node_id] = (reader, writer)
            return (reader, writer)
        except Exception as e:
            self.logger.error(f"Failed to connect to node {node.node_id}: {e}")
            return None
    
    async def _execute_local_task(self, task: DistributedTask) -> None:
        """Execute task locally.
        
        Args:
            task: Task to execute
        """
        start_time = time.time()
        
        try:
            task.status = TaskStatus.RUNNING
            self.logger.info(f"Executing task {task.task_id} locally")
            
            # Execute task based on type
            result = await self._dispatch_task_execution(task)
            
            # Task completed successfully
            task.result = result
            task.status = TaskStatus.COMPLETED
            
            execution_time = time.time() - start_time
            self.node_performance_history[self.node_info.node_id].append(execution_time)
            
            # Move to completed tasks
            if task.task_id in self.running_tasks:
                del self.running_tasks[task.task_id]
            self.completed_tasks[task.task_id] = task
            
            self.logger.info(f"Task {task.task_id} completed in {execution_time:.2f}s")
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Task {task.task_id} failed after {execution_time:.2f}s: {e}")
            
            task.error = str(e)
            task.status = TaskStatus.FAILED
            
            # Move to completed tasks (even if failed)
            if task.task_id in self.running_tasks:
                del self.running_tasks[task.task_id]
            self.completed_tasks[task.task_id] = task
        
        finally:
            # Update node load
            self.node_info.current_tasks = max(0, self.node_info.current_tasks - 1)
    
    async def _dispatch_task_execution(self, task: DistributedTask) -> Any:
        """Dispatch task execution based on task type.
        
        Args:
            task: Task to execute
            
        Returns:
            Task result
        """
        if task.task_type == "formalize_latex":
            return await self._execute_formalization_task(task)
        elif task.task_type == "verify_proof":
            return await self._execute_verification_task(task)
        elif task.task_type == "parse_latex":
            return await self._execute_parsing_task(task)
        else:
            raise ValueError(f"Unknown task type: {task.task_type}")
    
    async def _execute_formalization_task(self, task: DistributedTask) -> Dict[str, Any]:
        """Execute LaTeX formalization task."""
        # Mock implementation
        latex_content = task.payload.get('latex_content', '')
        target_system = task.payload.get('target_system', 'lean4')
        
        # Simulate processing time
        await asyncio.sleep(0.5)
        
        return {
            'formal_code': f"-- Mock {target_system} formalization\ntheorem mock : True := by trivial",
            'success': True,
            'processing_time': 0.5
        }
    
    async def _execute_verification_task(self, task: DistributedTask) -> Dict[str, Any]:
        """Execute proof verification task."""
        # Mock implementation
        formal_code = task.payload.get('formal_code', '')
        
        # Simulate verification time
        await asyncio.sleep(0.3)
        
        return {
            'verification_result': True,
            'errors': [],
            'verification_time': 0.3
        }
    
    async def _execute_parsing_task(self, task: DistributedTask) -> Dict[str, Any]:
        """Execute LaTeX parsing task."""
        # Mock implementation
        latex_content = task.payload.get('latex_content', '')
        
        # Simulate parsing time
        await asyncio.sleep(0.2)
        
        return {
            'theorems': [{'name': 'mock_theorem', 'statement': 'Mock theorem statement'}],
            'definitions': [],
            'parsing_time': 0.2
        }
    
    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """Handle incoming client connection."""
        client_addr = writer.get_extra_info('peername')
        self.logger.debug(f"Client connected from {client_addr}")
        
        try:
            while True:
                data = await reader.readline()
                if not data:
                    break
                
                try:
                    message = json.loads(data.decode().strip())
                    response = await self._process_message(message)
                    
                    response_data = json.dumps(response).encode() + b'\n'
                    writer.write(response_data)
                    await writer.drain()
                    
                except json.JSONDecodeError:
                    self.logger.error(f"Invalid JSON received from {client_addr}")
                    error_response = {'status': 'error', 'error': 'Invalid JSON'}
                    response_data = json.dumps(error_response).encode() + b'\n'
                    writer.write(response_data)
                    await writer.drain()
                    
        except Exception as e:
            self.logger.error(f"Error handling client {client_addr}: {e}")
        finally:
            writer.close()
            await writer.wait_closed()
            self.logger.debug(f"Client {client_addr} disconnected")
    
    async def _process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming message and return response."""
        msg_type = message.get('type')
        
        if msg_type == 'task_assignment':
            return await self._handle_task_assignment(message)
        elif msg_type == 'task_result':
            return await self._handle_task_result(message)
        elif msg_type == 'heartbeat':
            return await self._handle_heartbeat(message)
        elif msg_type == 'node_discovery':
            return await self._handle_node_discovery(message)
        else:
            return {'status': 'error', 'error': f'Unknown message type: {msg_type}'}
    
    async def _handle_task_assignment(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle task assignment message."""
        try:
            task_data = message['task']
            
            # Check if we can accept the task
            if self.node_info.current_tasks >= self.node_info.max_tasks:
                return {'status': 'rejected', 'error': 'Node at capacity'}
            
            # Create task
            task = DistributedTask(
                task_id=task_data['task_id'],
                task_type=task_data['task_type'],
                payload=task_data['payload'],
                priority=task_data.get('priority', 1),
                timeout=task_data.get('timeout', 300.0)
            )
            
            # Accept and execute task
            self.running_tasks[task.task_id] = task
            self.node_info.current_tasks += 1
            
            # Execute task asynchronously
            asyncio.create_task(self._execute_local_task(task))
            
            return {'status': 'accepted', 'task_id': task.task_id}
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    async def _handle_task_result(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle task result message."""
        # This would be used when remote nodes send back results
        return {'status': 'received'}
    
    async def _handle_heartbeat(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle heartbeat message."""
        node_id = message.get('node_id')
        if node_id and node_id in self.known_nodes:
            self.known_nodes[node_id].last_heartbeat = time.time()
            self.known_nodes[node_id].load_metrics = message.get('load_metrics', {})
            self.known_nodes[node_id].current_tasks = message.get('current_tasks', 0)
        
        return {
            'status': 'ok',
            'node_id': self.node_info.node_id,
            'timestamp': time.time()
        }
    
    async def _handle_node_discovery(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle node discovery message."""
        node_data = message.get('node_info')
        if node_data:
            node_info = NodeInfo(**node_data)
            self.known_nodes[node_info.node_id] = node_info
            self.logger.info(f"Discovered new node: {node_info.node_id}")
        
        return {
            'status': 'ok',
            'known_nodes': [
                {
                    'node_id': node.node_id,
                    'host': node.host,
                    'port': node.port,
                    'role': node.role.value
                }
                for node in self.known_nodes.values()
            ]
        }
    
    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeats to other nodes."""
        while True:
            try:
                # Update our own metrics
                self.node_info.load_metrics = await self._collect_load_metrics()
                self.node_info.last_heartbeat = time.time()
                
                # Send heartbeat to other nodes
                for node in self.known_nodes.values():
                    if node.node_id != self.node_info.node_id:
                        await self._send_heartbeat(node)
                
                await asyncio.sleep(self.config.heartbeat_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in heartbeat loop: {e}")
                await asyncio.sleep(self.config.heartbeat_interval)
    
    async def _collect_load_metrics(self) -> Dict[str, float]:
        """Collect current load metrics."""
        try:
            import psutil
            return {
                'cpu': psutil.cpu_percent(interval=1),
                'memory': psutil.virtual_memory().percent,
                'disk': psutil.disk_usage('/').percent
            }
        except ImportError:
            # Fallback if psutil not available
            return {'cpu': 0.0, 'memory': 0.0, 'disk': 0.0}
    
    async def _send_heartbeat(self, node: NodeInfo) -> None:
        """Send heartbeat to a specific node."""
        try:
            connection = await self._get_node_connection(node)
            if connection:
                reader, writer = connection
                
                message = {
                    'type': 'heartbeat',
                    'node_id': self.node_info.node_id,
                    'load_metrics': self.node_info.load_metrics,
                    'current_tasks': self.node_info.current_tasks,
                    'timestamp': time.time()
                }
                
                data = json.dumps(message).encode() + b'\n'
                writer.write(data)
                await writer.drain()
        except Exception as e:
            self.logger.debug(f"Failed to send heartbeat to {node.node_id}: {e}")
    
    async def _task_monitoring_loop(self) -> None:
        """Monitor running tasks for timeouts and failures."""
        while True:
            try:
                current_time = time.time()
                
                # Check for timed out tasks
                timed_out_tasks = []
                for task in self.running_tasks.values():
                    if current_time - task.created_at > task.timeout:
                        timed_out_tasks.append(task)
                
                # Handle timed out tasks
                for task in timed_out_tasks:
                    self.logger.warning(f"Task {task.task_id} timed out")
                    task.status = TaskStatus.FAILED
                    task.error = "Task timed out"
                    
                    # Move to completed
                    del self.running_tasks[task.task_id]
                    self.completed_tasks[task.task_id] = task
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in task monitoring loop: {e}")
                await asyncio.sleep(10)
    
    async def _node_discovery_loop(self) -> None:
        """Discover new nodes in the cluster."""
        while True:
            try:
                # Remove dead nodes
                current_time = time.time()
                dead_nodes = [
                    node_id for node_id, node in self.known_nodes.items()
                    if (node_id != self.node_info.node_id and 
                        current_time - node.last_heartbeat > self.config.node_timeout)
                ]
                
                for node_id in dead_nodes:
                    self.logger.warning(f"Removing dead node: {node_id}")
                    del self.known_nodes[node_id]
                    if node_id in self.client_connections:
                        reader, writer = self.client_connections[node_id]
                        writer.close()
                        del self.client_connections[node_id]
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in node discovery loop: {e}")
                await asyncio.sleep(30)
    
    async def _load_balancing_loop(self) -> None:
        """Periodically rebalance tasks across nodes."""
        while True:
            try:
                # Assign pending tasks
                for task in list(self.pending_tasks.values()):
                    if await self._assign_task(task):
                        break  # Process one task per iteration to avoid overwhelming
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in load balancing loop: {e}")
                await asyncio.sleep(5)
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """Get current cluster status."""
        return {
            'cluster_name': self.config.cluster_name,
            'node_count': len(self.known_nodes),
            'nodes': {
                node_id: {
                    'role': node.role.value,
                    'status': node.status,
                    'current_tasks': node.current_tasks,
                    'max_tasks': node.max_tasks,
                    'load_metrics': node.load_metrics,
                    'last_heartbeat': node.last_heartbeat
                }
                for node_id, node in self.known_nodes.items()
            },
            'tasks': {
                'pending': len(self.pending_tasks),
                'running': len(self.running_tasks),
                'completed': len(self.completed_tasks)
            },
            'this_node': {
                'node_id': self.node_info.node_id,
                'role': self.node_info.role.value,
                'current_tasks': self.node_info.current_tasks,
                'load_metrics': self.node_info.load_metrics
            }
        }