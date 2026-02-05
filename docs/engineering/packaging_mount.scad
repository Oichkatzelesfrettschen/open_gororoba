// Packaging Mount for Holographic Warp Gate
// Holds the 100mm Metasurface Chip and Fiber Couplers

$fn = 100;

module chip_holder() {
    difference() {
        // Main Block
        cube([120, 120, 10], center=true);
        
        // Chip Recess (100mm x 100mm x 1mm)
        translate([0, 0, 5])
        cube([100.5, 100.5, 2.5], center=true);
        
        // Through hole for optical path (90mm clear aperture)
        cylinder(h=20, d=90, center=true);
        
        // Screw Holes
        for(x=[-55, 55]) for(y=[-55, 55])
        translate([x, y, 0])
        cylinder(h=20, d=4, center=true);
    }
}

module fiber_coupler_mount() {
    translate([0, 0, 15])
    difference() {
        // Bar
        cube([120, 20, 10], center=true);
        
        // Fiber Port (SM1 Thread equivalent)
        rotate([90, 0, 0])
        cylinder(h=30, d=10, center=true);
    }
}

// Assembly
color("gray") chip_holder();
color("lightblue") fiber_coupler_mount();

// Simulated Chip
translate([0, 0, 4.5])
color("purple", 0.5)
cube([100, 100, 0.5], center=true);
