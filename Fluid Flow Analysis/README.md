# Fluid Flow Analysis

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Overview

**Fluid Flow Analysis** is a Python project that provides a set of tools for analyzing and classifying fluid flow, specifically focusing on calculating properties such as Reynolds number, flow type, skin friction, boundary layer thickness, and more. It is useful for engineers, researchers, and students working in fluid mechanics or related fields.

## Features

- **Reynolds Number Calculation:** Determine flow regime based on physical and fluid properties.
- **Flow Classification:** Classifies flow as Laminar, Transitional, or Turbulent.
- **Boundary Layer & Skin Friction:** Calculates boundary layer thickness and skin friction coefficients.
- **Shear Stress Calculation:** Computes wall shear stress for given flow conditions.
- **Kinetic Energy & Drag:** Estimates kinetic energy of flow and total skin friction drag.
- **Sphere Flow Description:** Provides qualitative description of wake behavior around a sphere based on Reynolds number.

## Getting Started

### Prerequisites

- Python 3.x

### Installation

Clone the repository:
```bash
git clone https://github.com/krish4210/helicopt.git
cd helicopt
```

No external dependencies are required beyond standard Python.

### Usage

You can run the script directly:
```bash
python "main.py"
```

The script uses predefined input values for fluid properties and outputs a summary of flow characteristics to the console.

#### Example Output

```
--- Basic Flow Classification ---
Reynolds Number: 150000.0
Flow Type: Laminar Flow
Sphere Flow Description: The flow is totally separated and the wake behind the sphere is large.

--- Incompressible Flow Properties ---
Flow Type: turbulent
Boundary Layer Thickness: 0.011698356023951062
Local Skin Friction Coefficient: 0.005434934260524957
Total Skin Friction Drag Coefficient: 0.006041375440671435
Total Skin Friction Drag: 0.01208275088134287
Shear Stress on Wall: 0.0006614378277661477
```

### Modifying Input Parameters

You can change the default fluid properties and flow parameters in the `main.py` file under the **Inputs** section to suit your analysis needs.

## Functions

The script includes the following main functions:

- `calculate_reynolds_number(density, velocity, length, viscosity)`
- `classify_flow(reynolds_number)`
- `describe_sphere_flow(reynolds_number)`
- `calculate_flow_type(reynolds_number)`
- `calculate_boundary_layer_thickness(reynolds_number, x)`
- `calculate_local_skin_friction_coefficient(reynolds_number)`
- `calculate_total_skin_friction_drag_coefficient(reynolds_number)`
- `calculate_ke_of_flow(density, velocity)`
- `calculate_reference_area(length, width)`
- `calculate_total_skin_friction_drag(drag_coefficient, area, ke)`
- `calculate_shear_stress_wall(reynolds_number)`

See the source code for detailed function documentation.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please open issues or submit pull requests for improvements.

## Author

[KrishChaudhari](https://github.com/KrishChaudhari)

---

Feel free to copy, modify, and expand this README as your project grows!