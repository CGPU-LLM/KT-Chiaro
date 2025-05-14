import sys, os
from typing import Dict, List, Optional, Union
import logging

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "ktransformers_ext", "build"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "ktransformers_ext", "build", "Release"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "ktransformers_ext", "build", "Debug"))
import cpuinfer_ext

logger = logging.getLogger(__name__)

class MoeTracker:
    """MoE层跟踪管理器，用于跟踪和管理模型中的MoE层状态"""
    
    def __init__(self):
        """初始化MoE层跟踪器"""
        # 初始化C++端的跟踪器
        cpuinfer_ext.moe_tracker_initialize()
        # 层名称到层ID的映射
        self.layer_name_to_id: Dict[str, int] = {}
        # 层ID到层名称的映射
        self.layer_id_to_name: Dict[int, str] = {}
        # 已注册的层
        self.registered_layers: List[str] = []
    
    def initialize(self):
        """重新初始化MoE层跟踪器"""
        cpuinfer_ext.moe_tracker_initialize()
        self.layer_name_to_id = {}
        self.layer_id_to_name = {}
        self.registered_layers = []
    
    def register_layer(self, layer_name: str) -> int:
        """
        注册一个MoE层
        
        Args:
            layer_name: 层的名称，通常是模型中的路径，如"model.layers.0.mlp"
            
        Returns:
            int: 注册的层ID
        """
        print(f'[MOE TRACKER] >>> register_layer: {layer_name}')
        if layer_name in self.layer_name_to_id:
            return self.layer_name_to_id[layer_name]
        
        # 在C++端注册
        layer_id = cpuinfer_ext.moe_tracker_register_layer(layer_name)
        print(f'>>> register_layer: {layer_name} with ID {layer_id}')
        # 更新本地映射
        self.layer_name_to_id[layer_name] = layer_id
        self.layer_id_to_name[layer_id] = layer_name
        self.registered_layers.append(layer_name)
        
        logger.debug(f"Registered MoE layer: {layer_name} with ID {layer_id}")
        return layer_id
    
    def get_layer_count(self) -> int:
        """获取已注册的MoE层数量"""
        return cpuinfer_ext.moe_tracker_get_layer_count()
    
    def set_current_layer(self, layer_id: int):
        """
        设置当前正在计算的MoE层
        
        Args:
            layer_id: 层ID
        """
        cpuinfer_ext.moe_tracker_set_current_layer(layer_id)
        
    def get_current_layer(self) -> int:
        """
        获取当前正在计算的MoE层ID
        
        Returns:
            int: 当前层ID，如果没有层在计算则返回-1
        """
        return cpuinfer_ext.moe_tracker_get_current_layer()
    
    def get_current_layer_name(self) -> Optional[str]:
        """
        获取当前正在计算的MoE层名称
        
        Returns:
            Optional[str]: 当前层名称，如果没有层在计算则返回None
        """
        layer_id = self.get_current_layer()
        if layer_id < 0:
            return None
        return cpuinfer_ext.moe_tracker_get_layer_name(layer_id)
    
    def get_layer_name(self, layer_id: int) -> Optional[str]:
        """
        获取指定层ID对应的层名称
        
        Args:
            layer_id: 层ID
            
        Returns:
            Optional[str]: 层名称，如果不存在则返回None
        """
        if layer_id in self.layer_id_to_name:
            return self.layer_id_to_name[layer_id]
        
        name = cpuinfer_ext.moe_tracker_get_layer_name(layer_id)
        if not name:
            return None
        
        self.layer_id_to_name[layer_id] = name
        return name
    
    def get_layer_id(self, layer_name: str) -> int:
        """
        获取指定层名对应的层ID
        
        Args:
            layer_name: 层名称
            
        Returns:
            int: 层ID，如果不存在则返回-1
        """
        if layer_name in self.layer_name_to_id:
            return self.layer_name_to_id[layer_name]
        
        layer_id = cpuinfer_ext.moe_tracker_get_layer_id_by_name(layer_name)
        if layer_id >= 0:
            self.layer_name_to_id[layer_name] = layer_id
            self.layer_id_to_name[layer_id] = layer_name
        
        return layer_id

# 全局单例实例
_moe_tracker = MoeTracker()

def get_moe_tracker() -> MoeTracker:
    """
    获取全局MoeTracker实例
    """
    return _moe_tracker 