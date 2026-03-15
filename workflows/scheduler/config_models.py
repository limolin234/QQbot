from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field

class StepConfig(BaseModel):
    action: str
    params: Dict[str, Any] = Field(default_factory=dict)

class ScheduleConfig(BaseModel):
    name: str
    type: Literal["cron", "interval"]
    expression: Optional[str] = None  # Required for cron
    seconds: Optional[int] = None     # Required for interval
    enabled: bool = True
    steps: List[StepConfig]

class SchedulerConfig(BaseModel):
    timezone: str = "Asia/Shanghai"
    schedules: List[ScheduleConfig] = Field(default_factory=list)
