def calculate_reynolds_number(density, velocity, length, viscosity):
    return (density * velocity * length) / viscosity

def classify_flow(reynolds_number):
    if reynolds_number < 3e5:
        return "Laminar Flow"
    elif 3e5 <= reynolds_number < 4e5:
        return "Transition Flow"
    else:
        return "Turbulent Flow"

def describe_sphere_flow(reynolds_number):
    if reynolds_number < 3e5:
        return "The flow is totally separated and the wake behind the sphere is large."
    elif 3e5 <= reynolds_number < 4e5:
        return "The separation point moves rearward and the wake behind the body is small."
    else:
        return "Flow is turbulent, wake is narrow and oscillatory."

def calculate_flow_type(reynolds_number):
    if reynolds_number < 5e5:
        return "laminar"
    elif 5e5 <= reynolds_number < 1e6:
        return "transition"
    else:
        return "turbulent"

def calculate_boundary_layer_thickness(reynolds_number, x):
    if reynolds_number <= 5e5:
        return 5.2 * x / reynolds_number**0.5
    else:
        return 0.37 * x / reynolds_number**0.2

def calculate_local_skin_friction_coefficient(reynolds_number):
    if reynolds_number <= 5e5:
        return 0.664 / reynolds_number**0.5
    else:
        return 0.0592 / reynolds_number**0.2

def calculate_total_skin_friction_drag_coefficient(reynolds_number):
    if reynolds_number <= 5e5:
        return 1.328 / reynolds_number**0.5
    else:
        return 0.074 / reynolds_number**0.2

def calculate_ke_of_flow(density, velocity):
    return 0.5 * density * velocity**2

def calculate_reference_area(length, width):
    return length * width

def calculate_total_skin_friction_drag(drag_coefficient, area, ke):
    return drag_coefficient * ke * area

def calculate_shear_stress_wall(reynolds_number):
    return 0.664 / reynolds_number**0.5

# Inputs
fluid_density = 1000.0  # kg/m^3
fluid_velocity = 1.5  # m/s
characteristic_length = 0.1  # m
fluid_viscosity = 0.001  # kg/(m*s)

freestream_velocity = 10.0  # m/s
reference_length_1 = 0.4  # m
reference_length = 1.0  # m
reference_width = 0.2  # m
freestream_density = 1.0  # kg/m^3
freestream_viscosity = 1e-5
x = reference_length_1

# Reynolds number calculations
reynolds_number_basic = calculate_reynolds_number(fluid_density, fluid_velocity, characteristic_length, fluid_viscosity)
reynolds_number_1 = calculate_reynolds_number(freestream_density, freestream_velocity, reference_length_1, freestream_viscosity)
reynolds_number = calculate_reynolds_number(freestream_density, freestream_velocity, reference_length, freestream_viscosity)

# Flow classification
flow_type_basic = classify_flow(reynolds_number_basic)
flow_type = calculate_flow_type(reynolds_number)
sphere_flow_description = describe_sphere_flow(reynolds_number_basic)

# Flow properties
boundary_layer_thickness = calculate_boundary_layer_thickness(reynolds_number, reference_length)
local_skin_friction_coefficient = calculate_local_skin_friction_coefficient(reynolds_number)
total_skin_friction_drag_coefficient = calculate_total_skin_friction_drag_coefficient(reynolds_number)
ke_of_flow = calculate_ke_of_flow(freestream_density, freestream_velocity)
reference_area = calculate_reference_area(reference_length, reference_width)
total_skin_friction_drag = calculate_total_skin_friction_drag(total_skin_friction_drag_coefficient, reference_area, ke_of_flow)
shear_stress_wall = calculate_shear_stress_wall(reynolds_number_1)

# Output
print("--- Basic Flow Classification ---")
print("Reynolds Number:", reynolds_number_basic)
print("Flow Type:", flow_type_basic)
print("Sphere Flow Description:", sphere_flow_description)

print("\n--- Incompressible Flow Properties ---")
print("Flow Type:", flow_type)
print("Boundary Layer Thickness:", boundary_layer_thickness)
print("Local Skin Friction Coefficient:", local_skin_friction_coefficient)
print("Total Skin Friction Drag Coefficient:", total_skin_friction_drag_coefficient)
print("Total Skin Friction Drag:", total_skin_friction_drag)
print("Shear Stress on Wall:", shear_stress_wall)