import configparser

from mxnet import nd

from lib.utils import scaled_Laplacian, cheb_polynomial, get_adjacency_matrix, get_adjacency_to_diffision_matrix


def get_backbones(config_filename, adj_filename, ctx):
    config = configparser.ConfigParser()
    config.read(config_filename)

    K = int(config['Training']['K'])
    num_of_weeks = int(config['Training']['num_of_weeks'])
    num_of_days = int(config['Training']['num_of_days'])
    num_of_hours = int(config['Training']['num_of_hours'])
    num_of_vertices = int(config['Data']['num_of_vertices'])

    # GDC Aware
    # adj_mx = get_adjacency_matrix(adj_filename, num_of_vertices)
    # for other non-diffision model, just adopt the "get_adjacency_matrix" to compute

    adj_mx = get_adjacency_to_diffision_matrix(adj_filename, num_of_vertices, 20)



    L_tilde = scaled_Laplacian(adj_mx)
    cheb_polynomials = [nd.array(i, ctx=ctx)
                        for i in cheb_polynomial(L_tilde, K)]

    backbones1 = [
        {
            "K": K,
            "num_of_chev_filters": 64,
            "num_of_time_filters": 64,
            "time_conv_strides": num_of_weeks,
            "cheb_polynomials": cheb_polynomials
        },
        {
            "K": K,
            "num_of_chev_filters": 64,
            "num_of_time_filters": 64,
            "time_conv_strides": 1,
            "cheb_polynomials": cheb_polynomials
        }
    ]

    backbones2 = [
        {
            "K": K,
            "num_of_chev_filters": 64,
            "num_of_time_filters": 64,
            "time_conv_strides": num_of_days,
            "cheb_polynomials": cheb_polynomials
        },
        {
            "K": K,
            "num_of_chev_filters": 64,
            "num_of_time_filters": 64,
            "time_conv_strides": 1,
            "cheb_polynomials": cheb_polynomials
        }
    ]

    backbones3 = [
        {
            "K": K,
            "num_of_chev_filters": 64,
            "num_of_time_filters": 64,
            "time_conv_strides": num_of_hours,
            "cheb_polynomials": cheb_polynomials
        },
        {
            "K": K,
            "num_of_chev_filters": 64,
            "num_of_time_filters": 64,
            "time_conv_strides": 1,
            "cheb_polynomials": cheb_polynomials
        }
    ]

    all_backbones = [
        backbones1,
        backbones2,
        backbones3
    ]

    return all_backbones
