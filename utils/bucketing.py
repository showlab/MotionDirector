from PIL import Image

def min_res(size, min_size): return 192 if size < 192 else size

def up_down_bucket(m_size, in_size, direction):
    if direction == 'down': return abs(int(m_size - in_size))
    if direction == 'up': return abs(int(m_size + in_size))

def get_bucket_sizes(size, direction: 'down', min_size):
    multipliers = [64, 128]
    for i, m in enumerate(multipliers):
        res =  up_down_bucket(m, size, direction)
        multipliers[i] = min_res(res, min_size=min_size)
    return multipliers

def closest_bucket(m_size, size, direction, min_size):
    lst = get_bucket_sizes(m_size, direction, min_size)
    return lst[min(range(len(lst)), key=lambda i: abs(lst[i]-size))]

def resolve_bucket(i,h,w): return  (i / (h / w))

def sensible_buckets(m_width, m_height, w, h, min_size=192):
    if h > w:
        w = resolve_bucket(m_width, h, w)
        w = closest_bucket(m_width, w, 'down', min_size=min_size)
        return w, m_height
    if h < w:
        h = resolve_bucket(m_height, w, h)
        h = closest_bucket(m_height, h, 'down', min_size=min_size)
        return m_width, h

    return m_width, m_height