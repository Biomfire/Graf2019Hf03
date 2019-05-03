//=============================================================================================
// Computer Graphics Sample Program: 3D engine-let
// Shader: Gouraud, Phong, NPR
// Material: diffuse + Phong-Blinn
// Texture: CPU-procedural
// Geometry: sphere, torus, mobius
// Camera: perspective
// Light: point
//=============================================================================================
#include "framework.h"

const int tessellationLevel = 30;
//https://stackoverflow.com/questions/1148309/inverting-a-4x4-matrix
bool gluInvertMatrix(const float m[16], float invOut[16]) {
    float inv[16], det;
    int i;

    inv[0] = m[5] * m[10] * m[15] -
             m[5] * m[11] * m[14] -
             m[9] * m[6] * m[15] +
             m[9] * m[7] * m[14] +
             m[13] * m[6] * m[11] -
             m[13] * m[7] * m[10];

    inv[4] = -m[4] * m[10] * m[15] +
             m[4] * m[11] * m[14] +
             m[8] * m[6] * m[15] -
             m[8] * m[7] * m[14] -
             m[12] * m[6] * m[11] +
             m[12] * m[7] * m[10];

    inv[8] = m[4] * m[9] * m[15] -
             m[4] * m[11] * m[13] -
             m[8] * m[5] * m[15] +
             m[8] * m[7] * m[13] +
             m[12] * m[5] * m[11] -
             m[12] * m[7] * m[9];

    inv[12] = -m[4] * m[9] * m[14] +
              m[4] * m[10] * m[13] +
              m[8] * m[5] * m[14] -
              m[8] * m[6] * m[13] -
              m[12] * m[5] * m[10] +
              m[12] * m[6] * m[9];

    inv[1] = -m[1] * m[10] * m[15] +
             m[1] * m[11] * m[14] +
             m[9] * m[2] * m[15] -
             m[9] * m[3] * m[14] -
             m[13] * m[2] * m[11] +
             m[13] * m[3] * m[10];

    inv[5] = m[0] * m[10] * m[15] -
             m[0] * m[11] * m[14] -
             m[8] * m[2] * m[15] +
             m[8] * m[3] * m[14] +
             m[12] * m[2] * m[11] -
             m[12] * m[3] * m[10];

    inv[9] = -m[0] * m[9] * m[15] +
             m[0] * m[11] * m[13] +
             m[8] * m[1] * m[15] -
             m[8] * m[3] * m[13] -
             m[12] * m[1] * m[11] +
             m[12] * m[3] * m[9];

    inv[13] = m[0] * m[9] * m[14] -
              m[0] * m[10] * m[13] -
              m[8] * m[1] * m[14] +
              m[8] * m[2] * m[13] +
              m[12] * m[1] * m[10] -
              m[12] * m[2] * m[9];

    inv[2] = m[1] * m[6] * m[15] -
             m[1] * m[7] * m[14] -
             m[5] * m[2] * m[15] +
             m[5] * m[3] * m[14] +
             m[13] * m[2] * m[7] -
             m[13] * m[3] * m[6];

    inv[6] = -m[0] * m[6] * m[15] +
             m[0] * m[7] * m[14] +
             m[4] * m[2] * m[15] -
             m[4] * m[3] * m[14] -
             m[12] * m[2] * m[7] +
             m[12] * m[3] * m[6];

    inv[10] = m[0] * m[5] * m[15] -
              m[0] * m[7] * m[13] -
              m[4] * m[1] * m[15] +
              m[4] * m[3] * m[13] +
              m[12] * m[1] * m[7] -
              m[12] * m[3] * m[5];

    inv[14] = -m[0] * m[5] * m[14] +
              m[0] * m[6] * m[13] +
              m[4] * m[1] * m[14] -
              m[4] * m[2] * m[13] -
              m[12] * m[1] * m[6] +
              m[12] * m[2] * m[5];

    inv[3] = -m[1] * m[6] * m[11] +
             m[1] * m[7] * m[10] +
             m[5] * m[2] * m[11] -
             m[5] * m[3] * m[10] -
             m[9] * m[2] * m[7] +
             m[9] * m[3] * m[6];

    inv[7] = m[0] * m[6] * m[11] -
             m[0] * m[7] * m[10] -
             m[4] * m[2] * m[11] +
             m[4] * m[3] * m[10] +
             m[8] * m[2] * m[7] -
             m[8] * m[3] * m[6];

    inv[11] = -m[0] * m[5] * m[11] +
              m[0] * m[7] * m[9] +
              m[4] * m[1] * m[11] -
              m[4] * m[3] * m[9] -
              m[8] * m[1] * m[7] +
              m[8] * m[3] * m[5];

    inv[15] = m[0] * m[5] * m[10] -
              m[0] * m[6] * m[9] -
              m[4] * m[1] * m[10] +
              m[4] * m[2] * m[9] +
              m[8] * m[1] * m[6] -
              m[8] * m[2] * m[5];

    det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];

    if (det == 0)
        return false;

    det = 1.0 / det;

    for (i = 0; i < 16; i++)
        invOut[i] = inv[i] * det;

    return true;
}
//---------------------------
struct Camera { // 3D camera
//---------------------------
	vec3 wEye, wLookat, wVup;   // extinsic
	float fov, asp, fp, bp;		// intrinsic
public:
	Camera() {
		asp = (float)windowWidth/windowHeight;
		fov = 75.0f * (float)M_PI / 180.0f;
		fp = 1; bp = 50;
	}
	mat4 V() { // view matrix: translates the center to the origin
		vec3 w = normalize(wEye - wLookat);
		vec3 u = normalize(cross(wVup, w));
		vec3 v = cross(w, u);
		return TranslateMatrix(wEye * (-1)) * mat4(u.x, v.x, w.x, 0,
											 u.y, v.y, w.y, 0,
											 u.z, v.z, w.z, 0,
											 0,   0,   0,   1);
	}
	mat4 P() { // projection matrix
		return mat4(1 / (tan(fov / 2)*asp), 0, 0, 0,
					0, 1 / tan(fov / 2), 0, 0,
					0, 0, -(fp + bp) / (bp - fp), -1,
					0, 0, -2 * fp*bp / (bp - fp), 0);
	}
	void Animate(float t) { }
};

//---------------------------
struct Material {
//---------------------------
	vec3 kd, ks, ka;
	float shininess;

	void SetUniform(unsigned shaderProg, char * name) {
		char buffer[256];
		sprintf(buffer, "%s.kd", name);
		kd.SetUniform(shaderProg, buffer);

		sprintf(buffer, "%s.ks", name);
		ks.SetUniform(shaderProg, buffer);

		sprintf(buffer, "%s.ka", name);
		ka.SetUniform(shaderProg, buffer);

		sprintf(buffer, "%s.shininess", name);
		int location = glGetUniformLocation(shaderProg, buffer);
		if (location >= 0) glUniform1f(location, shininess); else printf("uniform shininess cannot be set\n");
	}
};

//---------------------------
struct Light {
//---------------------------
	vec3 La, Le;
	vec4 wLightPos;
	vec4 position = vec4(0,0,20);

	void Animate(float t) {
	    vec4 q;
	    q.x = cos(t/4);
	    q.y = sin(t/4)*cos(t)/2;
	    q.z = sin(t/4)*sin(t)/2;
	    q.w = sin(t/4)*sqrt(3/4);
	    vec3 v = (position.x, position.y, position.z);
	    vec3 u(q.x, q.y, q.z);

        // Extract the scalar part of the quaternion
        float s = q.w;

        // Do the math
        vec3 pos = u * 2.0f * dot(u, v)
                 + v*(s*s - dot(u, u))
                 + cross(u, v)* 2.0f * s;
        wLightPos = vec4(pos.x, pos.y, pos.z, 0);
	}

	void SetUniform(unsigned shaderProg, char * name) {
		char buffer[256];
		sprintf(buffer, "%s.La", name);
		La.SetUniform(shaderProg, buffer);

		sprintf(buffer, "%s.Le", name);
		Le.SetUniform(shaderProg, buffer);

		sprintf(buffer, "%s.wLightPos", name);
		wLightPos.SetUniform(shaderProg, buffer);
	}
};

//---------------------------
struct CheckerBoardTexture : public Texture {
//---------------------------
	CheckerBoardTexture(const int width = 0, const int height = 0) : Texture() {
		glBindTexture(GL_TEXTURE_2D, textureId);    // binding
		std::vector<vec3> image(width * height);
		const vec3 yellow(1, 1, 0), blue(0, 0, 1);
		for (int x = 0; x < width; x++) for (int y = 0; y < height; y++) {
			image[y * width + x] = (x & 1) ^ (y & 1) ? yellow : blue;
		}
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_FLOAT, &image[0]); //Texture->OpenGL
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	}
};
//---------------------------
struct KaticaPottyok : public Texture {
//---------------------------
    KaticaPottyok(const int width = 0, const int height = 0) : Texture() {
        glBindTexture(GL_TEXTURE_2D, textureId);    // binding
        std::vector<vec3> image(width * height);
        const vec3 black(0, 0, 0), red(1, 0, 0);
        int Woffset = width/4;
        int Hoffset = height/4;
        int radius = width > height? height / 16 : width / 16;
        for (int x = 0; x < width; x++) for (int y = 0; y < height; y++) {
                    if(length(vec2(y,x)-vec2(1*Woffset,1*Hoffset))< 5 ||
                    length(vec2(y,x)-vec2(1*Woffset,2*Hoffset))< 5 ||
                    length(vec2(y,x)-vec2(1*Woffset,3*Hoffset))< 5 ||
                    length(vec2(y,x)-vec2(2*Woffset,2*Hoffset))< 5 ||
                    length(vec2(y,x)-vec2(3*Woffset,1*Hoffset))< 5 ||
                    length(vec2(y,x)-vec2(3*Woffset,2*Hoffset))< 5 ||
                    length(vec2(y,x)-vec2(3*Woffset,3*Hoffset))< 5) {
                        image[y * width + x] = black;
                    }
                    else
                        image[y *width + x] = red;
            }
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_FLOAT, &image[0]); //Texture->OpenGL
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    }
};
//---------------------------
struct RenderState {
//---------------------------
	mat4	           MVP, M, Minv, V, P;
	Material *         material;
	std::vector<Light> lights;
	Texture *          texture;
	vec3	           wEye;
};

//---------------------------
class Shader : public GPUProgram {
//--------------------------
public:
	virtual void Bind(RenderState state) = 0;
};

//---------------------------
class GouraudShader : public Shader {
//---------------------------
	const char * vertexSource = R"(
		#version 330
		precision highp float;

		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};

		struct Material {
			vec3 kd, ks, ka;
			float shininess;
		};

		uniform mat4  MVP, M, Minv;  // MVP, Model, Model-inverse
		uniform Light[8] lights;     // light source direction
		uniform int   nLights;
		uniform vec3  wEye;          // pos of eye
		uniform Material  material;  // diffuse, specular, ambient ref

		layout(location = 0) in vec3  vtxPos;            // pos in modeling space
		layout(location = 1) in vec3  vtxNorm;      	 // normal in modeling space

		out vec3 radiance;		    // reflected radiance

		void main() {
			gl_Position = vec4(vtxPos, 1) * MVP; // to NDC
			// radiance computation
			vec4 wPos = vec4(vtxPos, 1) * M;
			vec3 V = normalize(wEye * wPos.w - wPos.xyz);
			vec3 N = normalize((Minv * vec4(vtxNorm, 0)).xyz);
			if (dot(N, V) < 0) N = -N;	// prepare for one-sided surfaces like Mobius or Klein

			radiance = vec3(0, 0, 0);
			for(int i = 0; i < nLights; i++) {
				vec3 L = normalize(lights[i].wLightPos.xyz * wPos.w - wPos.xyz * lights[i].wLightPos.w);
				vec3 H = normalize(L + V);
				float cost = max(dot(N,L), 0), cosd = max(dot(N,H), 0);
				radiance += material.ka * lights[i].La + (material.kd * cost + material.ks * pow(cosd, material.shininess)) * lights[i].Le;
			}
		}
	)";

	// fragment shader in GLSL
	const char * fragmentSource = R"(
		#version 330
		precision highp float;

		in  vec3 radiance;      // interpolated radiance
		out vec4 fragmentColor; // output goes to frame buffer

		void main() {
			fragmentColor = vec4(radiance, 1);
		}
	)";
public:
	GouraudShader() { Create(vertexSource, fragmentSource, "fragmentColor"); }

	void Bind(RenderState state) {
		glUseProgram(getId()); 		// make this program run
		state.MVP.SetUniform(getId(), "MVP");
		state.M.SetUniform(getId(), "M");
		state.Minv.SetUniform(getId(), "Minv");
		state.wEye.SetUniform(getId(), "wEye");
		state.material->SetUniform(getId(), "material");

		int location = glGetUniformLocation(getId(), "nLights");
		if (location >= 0) glUniform1i(location, state.lights.size()); else printf("uniform nLight cannot be set\n");
		for (int i = 0; i < state.lights.size(); i++) {
			char buffer[256];
			sprintf(buffer, "lights[%d]", i);
			state.lights[i].SetUniform(getId(), buffer);
		}
	}
};

//---------------------------
class PhongShader : public Shader {
//---------------------------
	const char * vertexSource = R"(
		#version 330
		precision highp float;

		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};

		uniform mat4  MVP, M, Minv; // MVP, Model, Model-inverse
		uniform Light[8] lights;    // light sources
		uniform int   nLights;
		uniform vec3  wEye;         // pos of eye

		layout(location = 0) in vec3  vtxPos;            // pos in modeling space
		layout(location = 1) in vec3  vtxNorm;      	 // normal in modeling space
		layout(location = 2) in vec2  vtxUV;

		out vec3 wNormal;		    // normal in world space
		out vec3 wView;             // view in world space
		out vec3 wLight[8];		    // light dir in world space
		out vec2 texcoord;

		void main() {
			gl_Position = vec4(vtxPos, 1) * MVP; // to NDC
			// vectors for radiance computation
			vec4 wPos = vec4(vtxPos, 1) * M;
			for(int i = 0; i < nLights; i++) {
				wLight[i] = lights[i].wLightPos.xyz * wPos.w - wPos.xyz * lights[i].wLightPos.w;
			}
		    wView  = wEye * wPos.w - wPos.xyz;
		    wNormal = (Minv * vec4(vtxNorm, 0)).xyz;
		    texcoord = vtxUV;
		}
	)";

	// fragment shader in GLSL
	const char * fragmentSource = R"(
		#version 330
		precision highp float;

		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};

		struct Material {
			vec3 kd, ks, ka;
			float shininess;
		};

		uniform Material material;
		uniform Light[8] lights;    // light sources
		uniform int   nLights;
		uniform sampler2D diffuseTexture;

		in  vec3 wNormal;       // interpolated world sp normal
		in  vec3 wView;         // interpolated world sp view
		in  vec3 wLight[8];     // interpolated world sp illum dir
		in  vec2 texcoord;

        out vec4 fragmentColor; // output goes to frame buffer

		void main() {
			vec3 N = normalize(wNormal);
			vec3 V = normalize(wView);
			if (dot(N, V) < 0) N = -N;	// prepare for one-sided surfaces like Mobius or Klein
			vec3 texColor = texture(diffuseTexture, texcoord).rgb;
			vec3 ka = material.ka * texColor;
			vec3 kd = material.kd * texColor;

			vec3 radiance = vec3(0, 0, 0);
			for(int i = 0; i < nLights; i++) {
				vec3 L = normalize(wLight[i]);
				vec3 H = normalize(L + V);
				float cost = max(dot(N,L), 0), cosd = max(dot(N,H), 0);
				// kd and ka are modulated by the texture
				radiance += ka * lights[i].La + (kd * cost + material.ks * pow(cosd, material.shininess)) * lights[i].Le;
			}
			fragmentColor = vec4(radiance, 1);
		}
	)";
public:
	PhongShader() { Create(vertexSource, fragmentSource, "fragmentColor"); }

	void Bind(RenderState state) {
		glUseProgram(getId()); 		// make this program run
		state.MVP.SetUniform(getId(), "MVP");
		state.M.SetUniform(getId(), "M");
		state.Minv.SetUniform(getId(), "Minv");
		state.wEye.SetUniform(getId(), "wEye");
		state.material->SetUniform(getId(), "material");

		int location = glGetUniformLocation(getId(), "nLights");
		if (location >= 0) glUniform1i(location, state.lights.size()); else printf("uniform nLight cannot be set\n");
		for (int i = 0; i < state.lights.size(); i++) {
			char buffer[256];
			sprintf(buffer, "lights[%d]", i);
			state.lights[i].SetUniform(getId(), buffer);
		}
		state.texture->SetUniform(getId(), "diffuseTexture");
	}
};

//---------------------------
class NPRShader : public Shader {
//---------------------------
	const char * vertexSource = R"(
		#version 330
		precision highp float;

		uniform mat4  MVP, M, Minv; // MVP, Model, Model-inverse
		uniform	vec4  wLightPos;
		uniform vec3  wEye;         // pos of eye

		layout(location = 0) in vec3  vtxPos;            // pos in modeling space
		layout(location = 1) in vec3  vtxNorm;      	 // normal in modeling space
		layout(location = 2) in vec2  vtxUV;

		out vec3 wNormal, wView, wLight;				// in world space
		out vec2 texcoord;

		void main() {
		   gl_Position = vec4(vtxPos, 1) * MVP; // to NDC
		   vec4 wPos = vec4(vtxPos, 1) * M;
		   wLight = wLightPos.xyz * wPos.w - wPos.xyz * wLightPos.w;
		   wView  = wEye * wPos.w - wPos.xyz;
		   wNormal = (Minv * vec4(vtxNorm, 0)).xyz;
		   texcoord = vtxUV;
		}
	)";

	// fragment shader in GLSL
	const char * fragmentSource = R"(
		#version 330
		precision highp float;

		uniform sampler2D diffuseTexture;

		in  vec3 wNormal, wView, wLight;	// interpolated
		in  vec2 texcoord;
		out vec4 fragmentColor;    			// output goes to frame buffer

		void main() {
		   vec3 N = normalize(wNormal), V = normalize(wView), L = normalize(wLight);
		   float y = (dot(N, L) > 0.5) ? 1 : 0.5;
		   if (abs(dot(N, V)) < 0.2) fragmentColor = vec4(0, 0, 0, 1);
		   else						 fragmentColor = vec4(y * texture(diffuseTexture, texcoord).rgb, 1);
		}
	)";
public:
	NPRShader() { Create(vertexSource, fragmentSource, "fragmentColor"); }

	void Bind(RenderState state) {
		glUseProgram(getId()); 		// make this program run
		state.MVP.SetUniform(getId(), "MVP");
		state.M.SetUniform(getId(), "M");
		state.Minv.SetUniform(getId(), "Minv");
		state.wEye.SetUniform(getId(), "wEye");
		state.lights[0].wLightPos.SetUniform(getId(), "wLightPos");

		state.texture->SetUniform(getId(), "diffuseTexture");
	}
};

//---------------------------
struct VertexData {
//---------------------------
	vec3 position, normal;
	vec2 texcoord;
};

//---------------------------
class Geometry {
//---------------------------
protected:
	unsigned int vao;        // vertex array object
public:
	Geometry( ) {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);
	}
	virtual void Draw() = 0;
    virtual vec3 getCoords(float, float) = 0;
    virtual vec3 getNormal(float, float) = 0;
    virtual vec3 getRU(float, float) = 0;
    virtual vec3 getRV(float, float) = 0;
};

//---------------------------
class ParamSurface : public Geometry {
//---------------------------
	unsigned int nVtxPerStrip, nStrips;
public:
	ParamSurface() {
		nVtxPerStrip = nStrips = 0;
	}
	virtual VertexData GenVertexData(float u, float v) = 0;

	void Create(int N = tessellationLevel, int M = tessellationLevel) {
		unsigned int vbo;
		glGenBuffers(1, &vbo); // Generate 1 vertex buffer object
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		nVtxPerStrip = (M + 1) * 2;
		nStrips = N;
		std::vector<VertexData> vtxData;	// vertices on the CPU
		for (int i = 0; i < N; i++) {
			for (int j = 0; j <= M; j++) {
				vtxData.push_back(GenVertexData((float)j / M, (float)i / N));
				vtxData.push_back(GenVertexData((float)j / M, (float)(i + 1) / N));
			}
		}
		glBufferData(GL_ARRAY_BUFFER, nVtxPerStrip * nStrips * sizeof(VertexData), &vtxData[0], GL_STATIC_DRAW);
		// Enable the vertex attribute arrays
		glEnableVertexAttribArray(0);  // attribute array 0 = POSITION
		glEnableVertexAttribArray(1);  // attribute array 1 = NORMAL
		glEnableVertexAttribArray(2);  // attribute array 2 = TEXCOORD0
		// attribute array, components/attribute, component type, normalize?, stride, offset
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, position));
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, normal));
		glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, texcoord));
	}
	void Draw()	{
		glBindVertexArray(vao);
		for (int i = 0; i < nStrips; i++) glDrawArrays(GL_TRIANGLE_STRIP, i *  nVtxPerStrip, nVtxPerStrip);
	}
};

//---------------------------
struct Clifford {
//---------------------------
	float f, d;
	Clifford(float f0 = 0, float d0 = 0) { f = f0, d = d0; }
	Clifford operator+(Clifford r) { return Clifford(f + r.f, d + r.d); }
	Clifford operator-(Clifford r) { return Clifford(f - r.f, d - r.d); }
	Clifford operator*(Clifford r) { return Clifford(f * r.f, f * r.d + d * r.f); }
	Clifford operator/(Clifford r) {
		float l = r.f * r.f;
		return (*this) * Clifford(r.f / l, -r.d / l);
	}
};

Clifford T(float t) { return Clifford(t, 1); }
Clifford Sin(Clifford g) { return Clifford(sin(g.f), cos(g.f) * g.d); }
Clifford Cos(Clifford g) { return Clifford(cos(g.f), -sin(g.f) * g.d); }
Clifford Tan(Clifford g) { return Sin(g) / Cos(g); }
Clifford Log(Clifford g) { return Clifford(logf(g.f), 1 / g.f * g.d); }
Clifford Exp(Clifford g) { return Clifford(expf(g.f), expf(g.f) * g.d); }
Clifford Pow(Clifford g, float n) { return Clifford(powf(g.f, n), n * powf(g.f, n - 1) * g.d); }

class Klein : public ParamSurface{
public:
    Klein(){Create();};
    vec3 getCoords(float u, float v){
        float U = 2.0f * M_PI * fmod(u,1);
        float V = 2.0f * M_PI * fmod(v,1);
        //Calculate Position

        float a = cos(U) * (sin(U)+1) * 6;
        float b = sin(U) * 16;
        float c = ((cos(U)/-2)+1)*4;
        float x = M_PI < U && U <= M_PI*2 ? a + c * cos(V+ M_PI) : a + c * cos(U) * cos(V);
        float y = M_PI < U && U <= M_PI*2 ? b : b + c * sin(U) * cos(V);
        float z = c * sin(V);
        return vec3(x, y, z);
    }
    vec3 getRU(float u, float v){
        return normalize(getCoords(u, v)-getCoords(u-0.001, v));
    }
    vec3 getRV(float u, float v){
        return  normalize(getCoords(u, v)-getCoords(u, v-0.001));
    }
    vec3 getNormal(float u, float v){
        vec3 drdU = getRU(u,v);
        vec3 drdV = getRV(u,v);
        return normalize(cross(drdU, drdV));
    }
    VertexData GenVertexData(float u, float v) final{
        VertexData vd;
        //Calculate Position
        vd.position = getCoords(u, v);
        //Calculate Normal
        vd.normal = getNormal(u,v);
        //Calculate text coords
        vd.texcoord = vec2(u, v);
        return vd;
    }
};
class Dini : public ParamSurface{
    float a = 1;
    float b = 0.15;
public:
    Dini(){Create();};
    vec3 getCoords(float u, float v){
        float U = u * M_PI * 4;
        float V = v*0.99+0.01;
        float x = a * cos(U) * sin(V);
        float y = a * sin(U) * sin(V);
        float z = a * (cos(V) + log(tan(V/2))) + b*U;
        return vec3(x,y,z);
    }
    vec3 getRU(float u, float v){
        return normalize(getCoords(u, v)-getCoords(u-0.001, v));
    }
    vec3 getRV(float u, float v){
        return  normalize(getCoords(u, v)-getCoords(u, v-0.001));
    }
    vec3 getNormal(float u, float v){
        vec3 drdU = getRU(u,v);
        vec3 drdV = getRV(u,v);
        return normalize(cross(drdU, drdV));
    }
    VertexData GenVertexData(float u, float v) final{
        VertexData vd;
        vd.position = getCoords(u,v);
        vd.normal = getNormal(u, v);
        vd.texcoord = vec2(u, v);
        return vd;
    }
};
//---------------------------
class HalfElipsoid : public ParamSurface {
//---------------------------
public:
	HalfElipsoid() { Create(); }
    vec3 getCoords(float u, float v){
	    v +=1;
        return vec3(cos(u * 2.0f * M_PI) * sin(v/2*M_PI), sin(u*2.0f * M_PI) * sin(v/2*M_PI), cos(v/2*M_PI));
    }
    vec3 getRU(float u, float v){
        return normalize(getCoords(u, v)-getCoords(u-0.001, v));
    }
    vec3 getRV(float u, float v){
        return  normalize(getCoords(u, v)-getCoords(u, v-0.001));
    }
    vec3 getNormal(float u, float v){
        vec3 drdU = getRU(u,v);
        vec3 drdV = getRV(u,v);
        return normalize(cross(drdU, drdV));
    }
	VertexData GenVertexData(float u, float v) {
		VertexData vd;
		vd.position = vd.normal= getCoords(u,v);
		vd.texcoord = vec2(u, v);
		return vd;
	}
};
//---------------------------
struct Object {
//---------------------------
	Shader * shader;
	Material * material;
	Texture * texture;
	Geometry * geometry;
	vec3 scale, translation, rotationAxis;
	float rotationAngle;
public:
	Object(Shader * _shader, Material * _material, Texture * _texture, Geometry * _geometry) :
		scale(vec3(1, 1, 1)), translation(vec3(0, 0, 0)), rotationAxis(0, 0, 1), rotationAngle(0) {
		shader = _shader;
		texture = _texture;
		material = _material;
		geometry = _geometry;
	}

	virtual void Draw(RenderState state) {
		state.M = ScaleMatrix(scale) * RotationMatrix(rotationAngle, rotationAxis) * TranslateMatrix(translation);
		state.Minv = TranslateMatrix(-translation) * RotationMatrix(-rotationAngle, rotationAxis) * ScaleMatrix(vec3(1 / scale.x, 1 / scale.y, 1 / scale.z));
		state.MVP = state.M * state.V * state.P;
		state.material = material;
		state.texture = texture;
		shader->Bind(state);
		geometry->Draw();
	}

	virtual void Animate(float tstart, float tend) { rotationAngle += 0.8*(tend-tstart); }
};
struct SurfaceObject : public Object{
    vec2 dir = vec2(1,1);
    vec3 surfaceCoords = vec3(0,0,0);
    Object* surface;
    SurfaceObject(Object * _surface, Shader * _shader, Material * _material, Texture * _texture, Geometry * _geometry):surface(_surface), Object(_shader,_material, _texture,_geometry){};
    void Draw(RenderState state){
        vec3 i = surface->geometry->getRU(surfaceCoords.x, surfaceCoords.y)*dir.x;
        vec3 j = surface->geometry->getRV(surfaceCoords.x, surfaceCoords.y)*dir.y;
        vec3 N  = surface->geometry->getNormal(surfaceCoords.x, surfaceCoords.y);
        vec3 point = surface->geometry->getCoords(surfaceCoords.x, surfaceCoords.y);
        mat4 position = mat4(
                i.x,    i.y,    i.z,    0,
                j.x,    j.y,    j.z,    0,
                N.x,     N.y,     N.z,     0,
                point.x, point.y, point.z, 1);
        float floatpos[] = {i.x,    i.y,    i.z,    0,
                           j.x,    j.y,    j.z,    0,
                           N.x,     N.y,     N.z,     0,
                           point.x, point.y, point.z, 1};
        float invpos[16];
        gluInvertMatrix(floatpos, invpos);
        mat4 positioninv (  invpos[0],invpos[1],invpos[2],invpos[3],
                            invpos[4],invpos[5],invpos[6],invpos[7],
                            invpos[5],invpos[9],invpos[10],invpos[11],
                            invpos[6],invpos[13],invpos[14],invpos[15]);
        state.M = ScaleMatrix(scale)*RotationMatrix(rotationAngle, rotationAxis)*TranslateMatrix(translation)*position*ScaleMatrix(surface->scale)*RotationMatrix(surface->rotationAngle, surface->rotationAxis)*TranslateMatrix(surface->translation);
        state.Minv = TranslateMatrix(-surface->translation) * RotationMatrix(-surface->rotationAngle, surface->rotationAxis) * ScaleMatrix(vec3(1 / surface->scale.x, 1 / surface->scale.y, 1 / surface->scale.z)) * positioninv * TranslateMatrix(-translation) * RotationMatrix(-rotationAngle, rotationAxis) * ScaleMatrix(vec3(1 / scale.x, 1 / scale.y, 1 / scale.z));
        state.MVP = state.M * state.V * state.P;
        state.material = material;
        state.texture = texture;
        shader->Bind(state);
        geometry->Draw();
    }
    vec4 getRelCoords(vec3 relCoords = vec3(0,0,0)){
        vec3 Ru = surface->geometry->getRU(surfaceCoords.x, surfaceCoords.y);
        vec3 Rv = surface->geometry->getRV(surfaceCoords.x, surfaceCoords.y);
        vec3 N  = surface->geometry->getNormal(surfaceCoords.x, surfaceCoords.y);
        vec3 point = surface->geometry->getCoords(surfaceCoords.x, surfaceCoords.y);
        mat4 position = mat4(
                Ru.x,    Ru.y,    Ru.z,    0,
                Rv.x,    Rv.y,    Rv.z,    0,
                N.x,     N.y,     N.z,     0,
                point.x, point.y, point.z, 1);
        return vec4(relCoords.x,relCoords.y, relCoords.z,1)*ScaleMatrix(scale)*RotationMatrix(rotationAngle, rotationAxis)*TranslateMatrix(translation)*position*ScaleMatrix(surface->scale)*RotationMatrix(surface->rotationAngle, surface->rotationAxis)*TranslateMatrix(surface->translation);
    }
};
struct Katica: public SurfaceObject{
        Katica(Object * _surface, Shader * _shader, Material * _material, Texture * _texture, Geometry * _geometry):SurfaceObject(_surface,_shader,_material, _texture,_geometry)
        {
            rotationAxis = vec3(1,0,0);
            rotationAngle = M_PI;
        };
        static float anglewithU;
        float V = 0.1;
        static void AddAngle(){
            anglewithU += M_PI/4;
        };
        static void RemoveAngle(){
            anglewithU -= M_PI/4;
        };
        void Animate(float tstart, float tend){
            surfaceCoords.x += cos(anglewithU)*V*(tend-tstart);
            surfaceCoords.y += sin(anglewithU)*V*(tend-tstart);
        }

};
bool isClose = false;
float Katica::anglewithU = 0;
//---------------------------
class Scene {
//---------------------------
	std::vector<Object *> objects;
public:
	Camera camera; // 3D camera
	std::vector<Light> lights;
	Katica* katica;

	void Build() {
		// Shaders
		Shader * phongShader = new PhongShader();
		Shader * gouraudShader = new GouraudShader();
		Shader * nprShader = new NPRShader();

		// Materials
		Material * material0 = new Material;
		material0->kd = vec3(0.6f, 0.4f, 0.2f);
		material0->ks = vec3(4, 4, 4);
		material0->ka = vec3(0.1f, 0.1f, 0.1f);
		material0->shininess = 100;

		Material * material1 = new Material;
		material1->kd = vec3(0.8, 0.6, 0.4);
		material1->ks = vec3(0.3, 0.3, 0.3);
		material1->ka = vec3(0.2f, 0.2f, 0.2f);
		material1->shininess = 30;

		// Textures
		Texture * texture4x8 = new CheckerBoardTexture(4, 8);
		Texture * texture15x20 = new CheckerBoardTexture(15, 20);
		Texture * katicatext = new KaticaPottyok(50,50);

		// Geometries
		Geometry * klein = new Klein();
		Geometry * dini = new Dini();
		Geometry * halfelipsoid = new HalfElipsoid();

		// Create objects by setting up their vertex data on the GPU
		Object * sphereObject1 = new Object(phongShader, material0, texture4x8, klein);
		sphereObject1->translation = vec3(0, 1, 0);
		sphereObject1->rotationAxis = vec3(0, 1, 0);
		sphereObject1->scale = vec3(0.5,0.5,0.5);
		objects.push_back(sphereObject1);

        for(int i = 0; i < 30; i++) {
            SurfaceObject *noveny = new SurfaceObject(sphereObject1, phongShader, material0, texture4x8, dini);
            noveny->translation = vec3(0, 0, 3);
            noveny->surfaceCoords = vec3((float)(rand() % 100) / 100.0f, (float)(rand() % 100) /100.0f, 0);
            noveny->rotationAxis = vec3(0, 0, 1);
            noveny->scale = vec3(1.0, 1.0, 1.0);
            objects.push_back(noveny);
        }
        katica = new Katica(sphereObject1, phongShader, material0, katicatext, halfelipsoid);
        katica->surfaceCoords = vec3((float)(rand() % 100) / 100.0f, (float)(rand() % 100) /100.0f, 0);
        katica->scale = vec3(10.0/6, 6.0/6, 6.0/6);
        objects.push_back(katica);
		// Camera
		camera.wEye = vec3(0, 0, 30);
		camera.wLookat = vec3(0, 0, 0);
		camera.wVup = vec3(0, 1, 0);

		// Lights
		lights.resize(1);
		lights[0].wLightPos = vec4(0, 0, 4, 0);	// ideal point -> directional light source
		lights[0].La = vec3(0, 0, 0);
		lights[0].Le = vec3(3, 3, 3);

	}
	void Render() {
		RenderState state;
		state.wEye = camera.wEye;
		state.V = camera.V();
		state.P = camera.P();
		state.lights = lights;
		for (Object * obj : objects) obj->Draw(state);
		katica->Draw(state);
	}

	void Animate(float tstart, float tend) {
	    camera.wLookat = vec3(katica->getRelCoords().x, katica->getRelCoords().y, katica->getRelCoords().z);
	    vec3 cameraDiff = isClose? vec3(-5,0,-5):vec3(-20,0,-20);
        camera.wEye = vec3(katica->getRelCoords(cameraDiff).x, katica->getRelCoords(cameraDiff).y, katica->getRelCoords(cameraDiff).z);
		for (int i = 0; i < lights.size(); i++) { lights[i].Animate(tend); }
		for (Object * obj : objects) obj->Animate(tstart, tend);
		katica->Animate(tstart, tend);
	}
};

Scene scene;

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	glEnable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);
	scene.Build();
}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0.5f, 0.5f, 0.8f, 1.0f);							// background color
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the screen
	scene.Render();
	glutSwapBuffers();									// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
    switch(key){
        case 's':
            Katica::RemoveAngle();
            break;
        case 'a':
            Katica::AddAngle();
            break;
        case ' ':
            isClose = !isClose;
            break;
        default:
            break;
    }
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) { }

// Mouse click event
void onMouse(int button, int state, int pX, int pY) { }

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	static float tend = 0;
	const float dt = 0.1; // dt is infinitesimal
	float tstart = tend;
	tend = glutGet(GLUT_ELAPSED_TIME) / 1000.0f;

	for (float t = tstart; t < tend; t += dt) {
		float Dt = fmin(dt, tend - t);
		scene.Animate(t, t + Dt);
	}
	glutPostRedisplay();
}