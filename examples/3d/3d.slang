struct VSInput
{
    float4 pos;
    float4 col;
    float3 norm;
    float2 uv;
};

struct VSOutput
{
    float4 pos : SV_POSITION;
    float4 col;
    float3 norm;
};

struct PushConstants
{
    float4x4 mvp;
    float4 col;
};
[[push_constant]]
ConstantBuffer<PushConstants> push_constants;

[shader("vertex")]
VSOutput vertexMain(VSInput input)
{
    VSOutput output;
    // float4x4 mvp = mul(push_constants.model, mul(push_constants.view, push_constants.projection));
    float4x4 mvp = push_constants.mvp;
    float4 clipPos = mul(float4(input.pos.xyz, 1.0), mvp);
    output.pos = clipPos;
    output.norm = input.norm;
    // output.col = input.col;
    output.col = push_constants.col;
    // output.col = float4(debugColorFromID((uint)input.pos.w), 1.0);
    return output;
}

[shader("fragment")]
float4 fragmentMain(VSOutput input)
{
    float3 light_dir = normalize(float3(-0.5, -0.5, -0.5));
    float norm = dot(light_dir, input.norm) * 0.5 + 0.5;
    float4 c = pow(input.col, 1 / 2.2); // gamma correction for linear rendering
    c *= norm;
    c.a = 1;
    // c.xyz = input.norm;
    return c;
    // return float4(1.0, input.id / 8.0, 1.0, 1.0);
}

float3 debugColorFromID(uint id)
{
    // 8 distinct, high-contrast debug colors
    static const float3 colors[8] =
        {
            float3(1.0, 0.0, 0.0), // red
            float3(0.0, 1.0, 0.0), // green
            float3(0.0, 0.0, 1.0), // blue
            float3(1.0, 1.0, 0.0), // yellow
            float3(1.0, 0.0, 1.0), // magenta
            float3(0.0, 1.0, 1.0), // cyan
            float3(1.0, 0.5, 0.0), // orange
            float3(0.5, 0.0, 1.0)  // purple
        };

    return colors[id & 7]; // id % 8
}