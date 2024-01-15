### Universal AIMBOT for fps Video game

### 分为四个子模块

## 一、推理模块(Inference: onnx+yolov5+...)
AI模型保存为onnx格式，采用onnxruntime 进行推理，目前使用的推理模型是yolov5

## 二、鼠标移动模块(MouseEvent: GHub Driver)
解决调用什么接口进行移动。目前使用的是老版本的GHub驱动，可以达到伪装控制鼠标移动

## 三、桌面截取模块(ScreenCapture: dxgi+directX)
桌面截取使用dxgi截图速度非常快，但是这种截图会存在截取不到的问题、所以每次获取之后需要判断是否存在图像数据

## 四、移动算法模块(MoveAdaptor: PID)
解决鼠标如何移动、移动多少