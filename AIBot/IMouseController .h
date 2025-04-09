#include <cstdint>
#include <string>


enum class MouseMoveStatus {
    OK = 0,
    PLATFORM_NOT_SUPPORTED,  // The current platform is not supported
    PERMISSION_DENIED,       // Insufficient permissions (e.g., root required on Linux)
    DRIVER_FAILURE,          // Underlying driver/API call failed
    INVALID_INPUT            // Invalid input parameters
};


class IMouseController {
public:
    virtual ~IMouseController() = default;

    /**
     * Move the mouse pointer
     * @param delta_x    Relative movement on the X-axis (unit: pixels)
     * @param delta_y    Relative movement on the Y-axis (unit: pixels)
     * @param mode       Movement mode (default is relative movement)
     * @return           MouseError error code
     * @return           std::string detailed error message (optional)
     */
    virtual MouseMoveStatus MoveMouse(
        int delta_x, int delta_y
    ) = 0;

};