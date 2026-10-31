#include "GltfLoader.h"

#include <cstring>
#include <filesystem>
#include <iostream>
#include <glm/gtc/type_ptr.hpp>

#define CGLTF_IMPLEMENTATION
#include "cgltf.h"

// glTF factors and vertex colours are linear; ours are display values
static glm::vec3 ToDisplay(const glm::vec3& linear) {
    return glm::pow(glm::max(linear, glm::vec3(0.0f)), glm::vec3(1.0f / 2.2f));
}

static std::shared_ptr<Texture> LoadImage(const cgltf_image* image, const std::filesystem::path& dir) {
    auto tex = std::make_shared<Texture>();
    if (image->buffer_view) {   // embedded (.glb)
        const cgltf_buffer_view* view = image->buffer_view;
        const unsigned char* bytes = (const unsigned char*)view->buffer->data + view->offset;
        if (tex->LoadMemory(bytes, view->size)) return tex;
    } else if (image->uri && std::string(image->uri).rfind("data:", 0) != 0) {   // a file next to the .gltf
        std::string uri = image->uri;
        cgltf_decode_uri(&uri[0]);   // "%20" -> " "
        uri.resize(std::strlen(uri.c_str()));
        if (tex->LoadFile((dir / uri).string())) return tex;
    } else {
        std::cerr << "[GltfLoader] skipping an image stored as a data URI\n";
    }
    return nullptr;
}

static void AddPrimitive(const cgltf_primitive& prim, const glm::mat4& world,
                         const cgltf_data* data, MeshData& out) {
    if (prim.type != cgltf_primitive_type_triangles) return;

    const cgltf_accessor *pos = nullptr, *uv = nullptr, *col = nullptr;
    for (cgltf_size a = 0; a < prim.attributes_count; ++a) {
        const cgltf_attribute& attr = prim.attributes[a];
        if (attr.type == cgltf_attribute_type_position)                       pos = attr.data;
        else if (attr.type == cgltf_attribute_type_texcoord && attr.index == 0) uv = attr.data;
        else if (attr.type == cgltf_attribute_type_color && attr.index == 0)    col = attr.data;
    }
    if (!pos) return;

    // per-vertex arrays (cgltf_accessor_read_float handles every component type)
    const size_t n = pos->count;
    std::vector<glm::vec3> p(n), c(n, glm::vec3(1.0f));
    std::vector<glm::vec2> t(n, glm::vec2(0.0f));
    for (size_t i = 0; i < n; ++i) {
        float v[4] = { 0.0f, 0.0f, 0.0f, 1.0f };
        cgltf_accessor_read_float(pos, i, v, 3);
        p[i] = glm::vec3(world * glm::vec4(v[0], v[1], v[2], 1.0f));
        if (uv) {
            cgltf_accessor_read_float(uv, i, v, 2);
            t[i] = glm::vec2(v[0], v[1]);   // glTF's v = 0 is already the image top
        }
        if (col) {
            cgltf_accessor_read_float(col, i, v, cgltf_num_components(col->type));
            c[i] = ToDisplay(glm::vec3(v[0], v[1], v[2]));
        }
    }
    if (col) out.hasVertexColors = true;

    int material = prim.material ? (int)(prim.material - data->materials) : -1;
    size_t corners = prim.indices ? prim.indices->count : n;
    for (size_t k = 0; k + 2 < corners; k += 3) {
        size_t i[3];
        for (int j = 0; j < 3; ++j)
            i[j] = prim.indices ? cgltf_accessor_read_index(prim.indices, k + j) : k + j;
        if (i[0] >= n || i[1] >= n || i[2] >= n) continue;
        Triangle tri{p[i[0]], p[i[1]], p[i[2]]};
        tri.uv0 = t[i[0]]; tri.uv1 = t[i[1]]; tri.uv2 = t[i[2]];
        tri.c0  = c[i[0]]; tri.c1  = c[i[1]]; tri.c2  = c[i[2]];
        tri.material = material;
        out.tris.push_back(tri);
    }
}

bool GltfLoader::Load(const std::string& path, MeshData& out) {
    out = MeshData();
    cgltf_options options = {};
    cgltf_data* data = nullptr;
    if (cgltf_parse_file(&options, path.c_str(), &data) != cgltf_result_success) {
        std::cerr << "[GltfLoader] cannot parse " << path << "\n";
        return false;
    }
    if (cgltf_load_buffers(&options, data, path.c_str()) != cgltf_result_success) {
        std::cerr << "[GltfLoader] cannot load the buffers of " << path << "\n";
        cgltf_free(data);
        return false;
    }

    // Materials first, in glTF order, so a primitive's material pointer maps
    // to the same index. Images shared by several materials load once.
    const std::filesystem::path dir = std::filesystem::path(path).parent_path();
    std::vector<std::shared_ptr<Texture>> images(data->images_count);
    std::vector<bool> tried(data->images_count, false);
    for (cgltf_size m = 0; m < data->materials_count; ++m) {
        const cgltf_material& src = data->materials[m];
        Material mat;
        if (src.has_pbr_metallic_roughness) {
            const cgltf_pbr_metallic_roughness& pbr = src.pbr_metallic_roughness;
            mat.baseColor = ToDisplay(glm::vec3(pbr.base_color_factor[0], pbr.base_color_factor[1],
                                                pbr.base_color_factor[2]));
            const cgltf_texture* tex = pbr.base_color_texture.texture;
            if (tex && tex->image) {
                size_t img = tex->image - data->images;
                if (!tried[img]) {
                    images[img] = LoadImage(tex->image, dir);
                    tried[img] = true;
                }
                mat.texture = images[img];
            }
        }
        out.materials.push_back(mat);
    }

    // Every node of the scene that holds a mesh, at its world transform
    const cgltf_scene* scene = data->scene ? data->scene : (data->scenes_count ? &data->scenes[0] : nullptr);
    auto addNode = [&](const cgltf_node* node) {
        if (!node->mesh) return;
        float m[16];
        cgltf_node_transform_world(node, m);
        glm::mat4 world = glm::make_mat4(m);   // both column-major
        for (cgltf_size k = 0; k < node->mesh->primitives_count; ++k)
            AddPrimitive(node->mesh->primitives[k], world, data, out);
    };
    if (scene) {
        // walk the node trees under the scene's roots
        std::vector<const cgltf_node*> stack(scene->nodes, scene->nodes + scene->nodes_count);
        while (!stack.empty()) {
            const cgltf_node* node = stack.back();
            stack.pop_back();
            addNode(node);
            for (cgltf_size k = 0; k < node->children_count; ++k) stack.push_back(node->children[k]);
        }
    } else {
        for (cgltf_size k = 0; k < data->nodes_count; ++k) addNode(&data->nodes[k]);
    }

    cgltf_free(data);
    if (out.tris.empty()) {
        std::cerr << "[GltfLoader] no triangles in " << path << "\n";
        return false;
    }
    std::cout << "[GltfLoader] Loaded " << out.tris.size() << " triangles, "
              << out.materials.size() << " materials from " << path << "\n";
    return true;
}
