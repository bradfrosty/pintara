import Ember from 'ember';

export default Ember.Route.extend({
    actions: {
        recvFile: function(file) {
            console.log(file);
        }
    }
});
