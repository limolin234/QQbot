from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field


class StepConfig(BaseModel):
    action: str
    params: Dict[str, Any] = Field(default_factory=dict)


class StepTreeNode(BaseModel):
    kind: Literal["action", "group", "if"] = "action"

    # action node
    action: Optional[str] = None
    params: Dict[str, Any] = Field(default_factory=dict)

    # group node
    name: Optional[str] = None
    children: List["StepTreeNode"] = Field(default_factory=list)

    # if node
    condition: Dict[str, Any] = Field(default_factory=dict)
    then_steps: List["StepTreeNode"] = Field(default_factory=list)
    else_steps: List["StepTreeNode"] = Field(default_factory=list)


class ScheduleConfig(BaseModel):
    name: str
    type: Literal["cron", "interval"]
    expression: Optional[str] = None  # Required for cron
    seconds: Optional[int] = None  # Required for interval
    enabled: bool = True
    steps: List[StepConfig] = Field(default_factory=list)
    steps_tree: List[StepTreeNode] = Field(default_factory=list)


class SchedulerConfig(BaseModel):
    timezone: str = "Asia/Shanghai"
    schedules: List[ScheduleConfig] = Field(default_factory=list)


StepTreeNode.model_rebuild()
