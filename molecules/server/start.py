

if __name__ == "__main__":
    import sklearn
    assert sklearn.__version__ == '1.1.0', ('Exact sklearn version is needed because PyTDC uses pickled random forest'
                                            'that requires exact version match.')

    import sys
    #sys.path.append('/home/mateuszpyla/Pulpit/courses/AS/Automating_Science/molecules/')
    #print(sys.path)

    import os

    from server.routes import *
    from server.admin_routes import *
    from server import PORT
    # from port import PORT

    with app.app_context():
        db.create_all()
    # debug = False is key for PyTDC to work
    socketio.run(app, debug=False, port=PORT)
