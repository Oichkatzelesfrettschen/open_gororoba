// D3Q19 MRT LBM shader with BF16 storage precision.
//
// BF16 distribution storage via u32 bitcast (same as lbm_bf16.wgsl).
// MRT collision in FP32 (same as lbm_mrt.wgsl).
// Ghost moment damping (s_ghost=1.0) stabilizes BF16's 7-bit mantissa noise.

struct LbmConstants { nx: u32, ny: u32, nz: u32, gx: f32, gy: f32, gz: f32, }

@group(0) @binding(7) var<uniform> pc: LbmConstants;
@group(0) @binding(0) var<storage, read> f_in: array<u32>;
@group(0) @binding(1) var<storage, read_write> f_out: array<u32>;
@group(0) @binding(2) var<storage, read_write> rho_out: array<f32>;
@group(0) @binding(3) var<storage, read_write> u_out: array<f32>;
@group(0) @binding(4) var<storage, read> tau_in: array<f32>;
@group(0) @binding(5) var<storage, read> force_in: array<f32>;
@group(0) @binding(6) var<storage, read_write> entropy_out: array<f32>;

const CX = array<i32, 19>(0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0);
const CY = array<i32, 19>(0, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 1, -1, 1, -1);
const CZ = array<i32, 19>(0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1);
const WF = array<f32, 19>(
    0.33333333, 0.05555556, 0.05555556, 0.05555556, 0.05555556, 0.05555556, 0.05555556,
    0.02777778, 0.02777778, 0.02777778, 0.02777778, 0.02777778, 0.02777778,
    0.02777778, 0.02777778, 0.02777778, 0.02777778, 0.02777778, 0.02777778
);

fn bf16_to_f32(bits: u32) -> f32 { return bitcast<f32>(bits << 16u); }
fn f32_to_bf16(v: f32) -> u32 { return (bitcast<u32>(v) >> 16u); }

@compute @workgroup_size(8, 8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = id.x; let y = id.y; let z = id.z;
    if (x >= pc.nx || y >= pc.ny || z >= pc.nz) { return; }
    let idx = x + pc.nx * (y + pc.ny * z);
    let N = pc.nx * pc.ny * pc.nz;

    var rho = 0.0; var mx = 0.0; var my_v = 0.0; var mz_v = 0.0;
    var f: array<f32, 19>;
    for (var i = 0u; i < 19u; i++) {
        let val = bf16_to_f32(f_in[i * N + idx]);
        f[i] = max(0.0, val); rho += f[i];
        mx += f32(CX[i]) * f[i]; my_v += f32(CY[i]) * f[i]; mz_v += f32(CZ[i]) * f[i];
    }

    var ux = 0.0; var uy = 0.0; var uz = 0.0;
    if (rho > 1e-6) { let ir = 1.0 / rho; ux = mx * ir; uy = my_v * ir; uz = mz_v * ir;
        let speed = sqrt(ux*ux + uy*uy + uz*uz);
        if (speed > 0.15) { let s = 0.15/speed; ux *= s; uy *= s; uz *= s; }
    } else { rho = 1e-6; }

    rho_out[idx] = rho; u_out[idx] = ux; u_out[N+idx] = uy; u_out[2u*N+idx] = uz;

    // MRT collision
    let tau_val = max(0.51, tau_in[idx]);
    let s_nu = 1.0/tau_val; let s_e = 1.19; let s_eps = 1.4; let s_q = 1.2; let s_ghost = 1.0;
    let u_sq = ux*ux + uy*uy + uz*uz;

    let ax_p=f[1]+f[2]; let ax_m=f[1]-f[2]; let ay_p=f[3]+f[4]; let ay_m=f[3]-f[4];
    let az_p=f[5]+f[6]; let az_m=f[5]-f[6];
    let d78_p=f[7]+f[8]; let d78_m=f[7]-f[8]; let d910_p=f[9]+f[10]; let d910_m=f[9]-f[10];
    let d1112_p=f[11]+f[12]; let d1112_m=f[11]-f[12]; let d1314_p=f[13]+f[14]; let d1314_m=f[13]-f[14];
    let d1516_p=f[15]+f[16]; let d1516_m=f[15]-f[16]; let d1718_p=f[17]+f[18]; let d1718_m=f[17]-f[18];
    let axis_sum=ax_p+ay_p+az_p; let diag_sum=d78_p+d910_p+d1112_p+d1314_p+d1516_p+d1718_p;
    let xy_diag=d78_p+d910_p+d1112_p+d1314_p; let z_diag=d1516_p+d1718_p;
    let m0=f[0]+axis_sum+diag_sum;
    let m3=ax_m+d78_m+d910_m+d1112_m+d1314_m;
    let m5=ay_m+d78_m-d910_m+d1516_m+d1718_m;
    let m7=az_m+d1112_m-d1314_m+d1516_m-d1718_m;
    var m1=-30.0*f[0]-11.0*axis_sum+8.0*diag_sum;
    var m2=12.0*f[0]-4.0*axis_sum+diag_sum;
    var m4=-4.0*ax_m+d78_m+d910_m+d1112_m+d1314_m;
    var m6=-4.0*ay_m+d78_m-d910_m+d1516_m+d1718_m;
    var m8=-4.0*az_m+d1112_m-d1314_m+d1516_m-d1718_m;
    var m9=2.0*ax_p-(ay_p+az_p)+xy_diag-2.0*z_diag;
    var m10=-2.0*ax_p+(ay_p+az_p)+xy_diag-2.0*z_diag;
    var m11=ay_p-az_p+d78_p+d910_p-d1112_p-d1314_p;
    var m12=-ay_p+az_p+d78_p+d910_p-d1112_p-d1314_p;
    var m13=d78_p-d910_p; var m14=d1112_p-d1314_p; var m15=d1516_p-d1718_p;
    var m16=d78_m-d910_m-d1112_m+d1314_m;
    var m17=-d78_m-d910_m+d1516_m+d1718_m;
    var m18=d1112_m+d1314_m-d1516_m+d1718_m;
    let pxx=2.0*ux*ux-(uy*uy+uz*uz); let pww=uy*uy-uz*uz;
    m1-=s_e*(m1-rho*(19.0*u_sq-11.0)); m2-=s_eps*(m2-rho*(3.0-5.5*u_sq));
    m4-=s_q*(m4-(-2.0/3.0)*rho*ux); m6-=s_q*(m6-(-2.0/3.0)*rho*uy); m8-=s_q*(m8-(-2.0/3.0)*rho*uz);
    m9-=s_nu*(m9-rho*pxx); m10-=s_ghost*(m10-(-0.5)*rho*pxx);
    m11-=s_nu*(m11-rho*pww); m12-=s_ghost*(m12-(-0.5)*rho*pww);
    m13-=s_nu*(m13-rho*ux*uy); m14-=s_nu*(m14-rho*ux*uz); m15-=s_nu*(m15-rho*uy*uz);
    m16-=s_ghost*m16; m17-=s_ghost*m17; m18-=s_ghost*m18;

    let r0=m0*(1.0/19.0); let r1=m1*(1.0/2394.0); let r2=m2*(1.0/252.0);
    let r3=m3*(1.0/10.0); let r4=m4*(1.0/40.0); let r5=m5*(1.0/10.0); let r6=m6*(1.0/40.0);
    let r7=m7*(1.0/10.0); let r8=m8*(1.0/40.0);
    let r9=m9*(1.0/36.0); let r10=m10*(1.0/36.0); let r11=m11*(1.0/12.0); let r12=m12*(1.0/12.0);
    let r13=m13*0.25; let r14=m14*0.25; let r15=m15*0.25;
    let r16=m16*0.125; let r17=m17*0.125; let r18=m18*0.125;
    let bd=r0+r2; let rv=r9+r10; let rw=r11+r12;
    let s34=r3+r4; let s56=r5+r6; let s78=r7+r8;
    let ba=r0-11.0*r1-4.0*r2; let bxy=bd+rv+rw; let bxz=bd+rv-rw; let byz=bd-2.0*rv;

    f[0]=r0-30.0*r1+12.0*r2;
    f[1]=ba+r3-4.0*r4+2.0*r9-2.0*r10; f[2]=ba-r3+4.0*r4+2.0*r9-2.0*r10;
    f[3]=ba+r5-4.0*r6-r9+r10+r11-r12; f[4]=ba-r5+4.0*r6-r9+r10+r11-r12;
    f[5]=ba+r7-4.0*r8-r9+r10-r11+r12; f[6]=ba-r7+4.0*r8-r9+r10-r11+r12;
    {let p1=s34+s56;let n1=s34-s56;let p2=r13+r16;let n2=r13-r16;
     f[7]=8.0*r1+bxy+p1+p2-r17; f[8]=8.0*r1+bxy-p1+n2+r17;
     f[9]=8.0*r1+bxy+n1-p2-r17; f[10]=8.0*r1+bxy-n1-n2+r17;}
    {let p1=s34+s78;let n1=s34-s78;let p2=r14+r16;let n2=r14-r16;
     f[11]=8.0*r1+bxz+p1+n2+r18; f[12]=8.0*r1+bxz-p1+p2-r18;
     f[13]=8.0*r1+bxz+n1-n2+r18; f[14]=8.0*r1+bxz-n1-p2-r18;}
    {let p1=s56+s78;let n1=s56-s78;let p2=r15+r17;let n2=r15-r17;
     f[15]=8.0*r1+byz+p1+p2-r18; f[16]=8.0*r1+byz-p1+n2+r18;
     f[17]=8.0*r1+byz+n1-p2-r18; f[18]=8.0*r1+byz-n1-n2+r18;}

    // Streaming with BF16 store
    let nx_i=i32(pc.nx); let ny_i=i32(pc.ny); let nz_i=i32(pc.nz);
    for (var i = 0u; i < 19u; i++) {
        let next_x=(i32(x)+CX[i]+nx_i)%nx_i; let next_y=(i32(y)+CY[i]+ny_i)%ny_i;
        let next_z=(i32(z)+CZ[i]+nz_i)%nz_i;
        let next_idx=u32(next_x)+pc.nx*(u32(next_y)+pc.ny*u32(next_z));
        f_out[i*N+next_idx] = f32_to_bf16(f[i]);
    }
    entropy_out[idx] = 0.0;
}
