#include <Mouse.h>

// Signed char can be between -128 to 127
int delta[2];
int negMax = -127;
int posMax = 127;

#define COM_END_CHAR '\n'

enum class CommandType : int32_t {
  CMD_INIT_DEVICE = 0x1,
  CMD_MOUSE_MOVE = 0x2,
  CMD_CLOSE_DEVICE = 0x3,
};

struct COMPackage {
  enum CommandType type;
  
  union PackageContent {
    struct
    {
      int32_t padding;
    } init_arg;

    struct
    {
      int32_t delta_x;
      int32_t delta_y;
    } move_arg;

    struct
    {
      int32_t padding;
    } close_arg;
  } Content;

  void InitMoveArg(int32_t delta_x, int32_t delta_y) {
    Content.move_arg.delta_x = delta_x;
    Content.move_arg.delta_y = delta_y;
  }

  void InitDeviceArg(int32_t arg) {
    Content.init_arg.padding = arg;
  }
};

void setup() {
  Mouse.begin();
  Serial.begin(115200);
}

void handleMouseMove(int32_t delta_x, int32_t delta_y)
{
  handleX(delta_x);
  handleY(delta_y);
}

void loop() {
  if (Serial.available() > 0) {
    COMPackage pkg;

    size_t size = Serial.readBytes(reinterpret_cast<char*>(&pkg), sizeof(COMPackage));

    if (size != sizeof(pkg)) return;

    if (pkg.type == CommandType::CMD_MOUSE_MOVE) {
      handleMouseMove(pkg.Content.move_arg.delta_x, pkg.Content.move_arg.delta_y);
    }
    // something else control
  }
}

// Handle Moving of x
void handleX(int dx) {
  int spawns;
  int remainder;

  if (dx < negMax) {
    spawns = int(dx / negMax);
    remainder = int(dx % negMax);

    for (int i = 0; i < spawns; i++) {
      Mouse.move(negMax, 0, 0);
    }
    Mouse.move(remainder, 0, 0);
  } else if (dx >= negMax && dx <= posMax) {
    Mouse.move(dx, 0, 0);
  } else if (dx > posMax) {
    spawns = int(dx / posMax);
    remainder = int(dx % posMax);

    for (int i = 0; i < spawns; i++) {
      Mouse.move(posMax, 0, 0);
    }
    Mouse.move(remainder, 0, 0);
  }
}

// Handle Moving of y
void handleY(int dy) {
  int spawns;
  int remainder;
  if (dy < negMax) {
    spawns = int(dy / negMax);
    remainder = int(dy % negMax);
    remainder *= -1;
    for (int i = 0; i < spawns; i++) {
      Mouse.move(0, posMax, 0);
    }
    Mouse.move(0, remainder, 0);
  } else if (dy >= negMax && dy <= posMax) {
    dy *= -1;
    Mouse.move(0, dy, 0);
  } else if (dy > posMax) {
    spawns = int(dy / posMax);
    remainder = int(dy % posMax);
    remainder *= -1;
    for (int i = 0; i < spawns; i++) {
      Mouse.move(0, negMax, 0);
    }
    Mouse.move(0, remainder, 0);
  }
}