import Ember from 'ember';

export default Ember.Component.extend({
    tagName: 'div',
    classNameBindings: 'isDisabled:is-disabled'.w(),
    attributeBindings: 'data-uploader'.w(),
    'data-uploader': 'true',
    isDisabled: false,
    dragCount: 0,
    isDragging: Ember.computed('dragCount', function() {
        return this.get('dragCount') > 0;
    }),
    displayText: "Drag here",

    dragOver: function(event) {
        // this is needed to avoid the default behaviour from the browser
        event.preventDefault();
    },

    dragEnter: function(event) {
        event.preventDefault();
        this.incrementProperty('dragCount');
    },

    dragLeave: function(event) {
        event.preventDefault();
        this.decrementProperty('dragCount');
    },

    drop: function(event) {
        var file;

        if(!this.get('isDisabled')){
            event.preventDefault();
            this.set('isDragging', false);

            // only 1 file for now
            file = event.dataTransfer.files[0];
            // this.set('isDisabled', true);
            this.sendAction('dropFile', file);
        } else{
            console.error('you can only upload on file at the time');
        }
    }
});
