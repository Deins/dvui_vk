struct VSInput
{
    float4 pos;
};

struct VSOutput
{
    float4 pos : SV_POSITION;
    float4 col;
};

struct PushConstants
{
    float4x4 model;
    float4x4 view;
    float4x4 projection;
};
[[push_constant]]
ConstantBuffer<PushConstants> push_constants;

[shader("vertex")]
VSOutput vertexMain(VSInput input)
{
    VSOutput output;
    float4x4 mvp = mul(push_constants.model, mul(push_constants.view, push_constants.projection));
    float4 clipPos = mul(float4(input.pos.xyz, 1.0), mvp);
    output.pos = clipPos;
    output.col = float4(debugColorFromID((uint)input.pos.w), 1.0);
    return output;
}

[shader("fragment")]
float4 fragmentMain(VSOutput input)
{
    return input.col;
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